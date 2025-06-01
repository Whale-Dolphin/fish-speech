import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config
from loguru import logger
from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.content_sequence import ContentSequence
from fish_speech.conversation import Conversation, Message
from fish_speech.schema import (
    ServeChatRequest,
    ServeChatResponse,
    ServeCompletionRequest,
    ServeCompletionResponse,
    ServeContentSequenceParts,
    ServeContentSequenceRequest,
    ServeContentSequenceResponse,
    ServeMessage,
    ServeStreamContentSequenceDelta,
    ServeStreamContentSequenceResponse,
    ServeStreamDelta,
    ServeStreamResponse,
    ServeTextPart,
    ServeVQPart,
)
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from fish_speech.models.text2semantic.llama import (
    BaseModelArgs,
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[:, -1],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    audio_masks: torch.Tensor = None,
    audio_parts: torch.Tensor = None,
) -> torch.Tensor:
    # print(x, torch.count_nonzero(vq_masks))
    x = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]

    codebooks = [
        sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[:, 0] if previous_tokens is not None else None
            ),  # Disable repetition penalty for the token codebook
        )[0]
    ]

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits[:, :, :1024]

        # Convert logits to probs
        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[:, codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
        )[0]

        # print(codebook_idx, a)
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    # mask = (codebooks[:, :1, :] >= model.tokenizer.semantic_begin_id) & (codebooks[:, :1, :] <= model.tokenizer.semantic_end_id)
    # codebooks[:, 1:, :] = torch.masked_fill(
    #     codebooks[:, 1:, :], ~mask, CODEBOOK_PAD_TOKEN_ID
    # )

    # for i in range(codebooks.size(1) - 1):
    #     codebooks[:, i + 1, :] = torch.masked_fill(
    #         codebooks[:, i + 1, :],
    #         codebooks[:, :1, :] != semantic_id,
    #         CODEBOOK_PAD_TOKEN_ID + i * 1024,
    #     )

    # print(codebooks)

    return codebooks


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_ar,
    early_stop_threshold: float = 0.6,
):
    batch_size = cur_token.size(0)
    previous_tokens = torch.zeros(
        (batch_size, model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=cur_token.device)
    finished = finished | (cur_token[:, 0, -1] == im_end_id)
    start_time = time.time()

    for i in range(num_new_tokens):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :, :win_size]
        else:
            window = previous_tokens[:, :, i - win_size : i]

        with sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(batch_size, model.config.num_codebooks + 1, -1)
        previous_tokens[:, :, i : i + 1] = next_token.view(
            batch_size, model.config.num_codebooks + 1, -1
        )

        yield cur_token.cpu()

        finished = finished | (cur_token[:, 0, -1] == im_end_id)
        if finished.all() or (
            0 < early_stop_threshold < 1
            and finished.sum() >= round(batch_size * early_stop_threshold)
        ):
            break

    total_time = time.time() - start_time
    generated_tokens = i + 1
    tokens_per_second = (generated_tokens / total_time) * batch_size
    logger.info(
        f"Decoded {generated_tokens} x {batch_size} tokens in {total_time * 1000:.2f}ms ({tokens_per_second:.2f} tokens/s)"
    )


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: BaseTransformer,
    prompt: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    max_new_tokens: int,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_ar,
    num_samples: int = 1,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(
            max_batch_size=num_samples,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device)

    temperature = torch.tensor(
        [[sampling_kwargs["temperature"]]], device=device, dtype=torch.bfloat16
    )
    top_p = torch.tensor(
        [[sampling_kwargs["top_p"]]], device=device, dtype=torch.bfloat16
    )
    repetition_penalty = torch.tensor(
        [[sampling_kwargs["repetition_penalty"]]], device=device, dtype=torch.bfloat16
    )

    # Use non-accelerated version for now, to avoid compilation overhead
    next_token = (
        decode_one_token_ar(
            model,
            prompt,
            input_pos,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
        )
        .clone()
        .view(num_samples, codebook_dim, -1)
    )
    yield next_token.cpu()

    temperature = temperature.repeat(num_samples, 1)
    top_p = top_p.repeat(num_samples, 1)
    repetition_penalty = repetition_penalty.repeat(num_samples, 1)
    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    yield from decode_n_tokens(
        model,
        next_token,
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        im_end_id=im_end_id,
        decode_one_token=decode_one_token,
        early_stop_threshold=early_stop_threshold,
    )


def load_model(checkpoint_path, device, precision, compile=False):
    model: Union[NaiveTransformer, DualARTransformer] = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True
    )

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        prefill_n_tokens = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        raise ValueError("Unsupported model type")

    # Initialize cache
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            # mode="max-autotune-no-cudagraphs",
            mode="reduce-overhead",
            fullgraph=True,
        )

    return model.eval(), decode_one_token


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


@dataclass
class SwapModelRequest:
    state_dict: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    tokenizer = FishTokenizer.from_pretrained(checkpoint_path)
    config = BaseModelArgs.from_pretrained(checkpoint_path)

    def worker():
        model, decode_one_token = load_model(
            checkpoint_path, device, precision, compile=compile
        )
        init_event.set()

        while True:
            item: GenerateRequest | SwapModelRequest | None = input_queue.get()
            if item is None:
                break

            if isinstance(item, SwapModelRequest):
                device_state_dict = {
                    k: v.to(device) for k, v in item.state_dict.items()
                }
                logger.info(f"Swapping model weights")
                try:
                    result = model.load_state_dict(
                        device_state_dict, strict=False, assign=True
                    )
                    assert (
                        len(result.unexpected_keys) == 0
                    ), f"Unexpected keys: {result.unexpected_keys}"
                    missing_keys = set(result.missing_keys)
                    for k in result.missing_keys:
                        if "kv_cache" in k:
                            missing_keys.remove(k)
                    assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
                    logger.info(f"Model weights swapped")
                    item.response_queue.put(None)
                except Exception as e:
                    logger.exception(f"Error loading state dict: {e}")
                    item.response_queue.put(str(e))
                continue

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for token in generate(
                    model=model,
                    decode_one_token=decode_one_token,
                    **kwargs,
                ):
                    response_queue.put(token)

                response_queue.put("stop")
            except Exception as e:
                import traceback

                logger.exception(f"Error in worker: {traceback.format_exc()}")
                response_queue.put("error")

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue, tokenizer, config


def execute_chat_request(
    input_queue: queue.Queue,
    tokenizer: FishTokenizer,
    config: BaseModelArgs,
    request: ServeChatRequest,
    device="cuda:0",
):
    im_end_id = tokenizer.get_token_id(IM_END_TOKEN)

    messages = []
    for message in request.messages:
        messages.append(message.to_conversation_message())

    assert len(messages) >= 1, "At least one message is required"
    # assert messages[-1].role == "user", "The last message must be from the user"

    if messages[-1].role == "user":
        messages.append(
            Message(role="assistant", parts=[], add_im_end=False, modality="voice")
        )
    elif messages[-1].role == "raw":
        messages[-1].add_im_start = False
        messages[-1].add_im_end = False
        messages[-1].modality = "voice"
    else:
        assert (
            messages[-1].role == "assistant"
        ), "The last message must be from the assistant"
        messages[-1].add_im_end = False

    conv = Conversation(messages=messages)
    # conv.visualize(tokenizer)

    prompt, audio_masks, audio_parts = conv.encode_for_inference(
        tokenizer=tokenizer, num_codebooks=config.num_codebooks
    )
    prompt = prompt.to(device)
    audio_masks = audio_masks.to(device) if audio_masks is not None else None
    audio_parts = (
        audio_parts.to(device, dtype=torch.bfloat16)
        if audio_parts is not None
        else None
    )

    if request.streaming:
        for i in range(request.num_samples):
            yield ServeStreamResponse(
                sample_id=i,
                delta=ServeStreamDelta(
                    role="assistant",
                ),
            )

    req = {
        "prompt": prompt,
        "audio_masks": audio_masks,
        "audio_parts": audio_parts,
        "max_new_tokens": request.max_new_tokens,
        "im_end_id": im_end_id,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "repetition_penalty": request.repetition_penalty,
        "num_samples": request.num_samples,
        "early_stop_threshold": request.early_stop_threshold,
    }

    start = time.time()
    response_queue = queue.Queue()
    input_queue.put(GenerateRequest(req, response_queue))

    # Decoding
    decode_buffer = [[] for _ in range(request.num_samples)]
    parts = [[] for _ in range(request.num_samples)]

    def send_reset_buffer(sample_id):
        nonlocal decode_buffer
        if len(decode_buffer[sample_id]) == 0:
            return

        decoded = tokenizer.decode(decode_buffer[sample_id])
        part = ServeTextPart(text=decoded)

        if request.streaming:
            yield ServeStreamContentSequenceResponse(
                sample_id=sample_id, delta=ServeStreamContentSequenceDelta(part=part)
            )
        else:
            parts[sample_id].append(part)

        decode_buffer[sample_id] = []

    # Decode process
    finished = [False for _ in range(request.num_samples)]
    stats = {}
    idx = 0
    while True:
        response = response_queue.get()

        if response in ["stop", "error"]:
            break

        for sample_id, tokens in enumerate(response):
            if finished[sample_id]:
                continue

            if tokens[0] == im_end_id:
                finished[sample_id] = True
                if request.streaming:
                    yield from send_reset_buffer(sample_id)
                    yield ServeStreamContentSequenceResponse(
                        sample_id=sample_id,
                        finish_reason="stop",
                        stats=stats,
                    )
                continue

            is_semantic = (
                tokenizer.semantic_begin_id <= tokens[0] <= tokenizer.semantic_end_id
            )
            if is_semantic and request.streaming:
                yield from send_reset_buffer(sample_id)
                # Streaming vq
                _tokens = tokens[1:].clone()

                if config.share_codebook_embeddings is False:
                    for i in range(len(_tokens)):
                        _tokens[i] -= config.codebook_size * i

                yield ServeStreamContentSequenceResponse(
                    sample_id=sample_id,
                    delta=ServeStreamContentSequenceDelta(
                        part=ServeVQPart(codes=_tokens.tolist())
                    ),
                )
                continue

            # Not streaming vq
            if is_semantic:
                yield from send_reset_buffer(sample_id)
                # None streaming vq
                if len(parts[sample_id]) == 0 or not isinstance(
                    parts[sample_id][-1], ServeVQPart
                ):
                    _tokens = tokens[1:].clone()

                    if config.share_codebook_embeddings is False:
                        for i in range(len(_tokens)):
                            _tokens[i] -= config.codebook_size * i

                    parts[sample_id].append(ServeVQPart(codes=_tokens.tolist()))
                else:
                    for codebook_id, value in enumerate(tokens[1:, :]):
                        val = value.item()
                        if config.share_codebook_embeddings is False:
                            val -= config.codebook_size * codebook_id

                        parts[sample_id][-1].codes[codebook_id].append(val)
                continue

            if not is_semantic:
                # Stream text decode is not supported now
                decode_buffer[sample_id].append(tokens[0, 0])

        if idx == 0:
            stats["time_to_first_token"] = (time.time() - start) * 1000

        idx += 1

    for sample_id in range(request.num_samples):
        yield from send_reset_buffer(sample_id)

    stats["total_time"] = (time.time() - start) * 1000
    stats["total_tokens"] = idx

    if request.streaming:
        for sample_id in range(request.num_samples):
            if finished[sample_id]:
                continue
            yield ServeStreamContentSequenceResponse(
                finish_reason=response, stats=stats, sample_id=sample_id
            )
        return

    yield ServeChatResponse(
        messages=[
            ServeMessage(role="assistant", parts=parts[i])
            for i in range(request.num_samples)
        ],
        finish_reason=response,
        stats=stats,
    )


def execute_content_sequence_request(
    input_queue: queue.Queue,
    tokenizer: FishTokenizer,
    config: BaseModelArgs,
    request: ServeContentSequenceRequest,
    device="cuda:0",
):
    im_end_id = tokenizer.get_token_id(IM_END_TOKEN)

    # Create ContentSequence from the provided content dict
    seq = ContentSequence(**request.content)

    # seq.visualize(tokenizer)

    # Encode the sequence for inference
    values, audio_masks, audio_parts = seq.encode_for_inference(
        tokenizer, config.num_codebooks
    )

    # Move tensors to the right device
    prompt = values.to(device)
    audio_masks = audio_masks.to(device) if audio_masks is not None else None
    audio_parts = (
        audio_parts.to(device, dtype=torch.bfloat16)
        if audio_parts is not None
        else None
    )

    req = {
        "prompt": prompt,
        "audio_masks": audio_masks,
        "audio_parts": audio_parts,
        "max_new_tokens": request.max_new_tokens,
        "im_end_id": im_end_id,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "repetition_penalty": request.repetition_penalty,
        "num_samples": request.num_samples,
        "early_stop_threshold": request.early_stop_threshold,
    }

    start = time.time()
    response_queue = queue.Queue()
    input_queue.put(GenerateRequest(req, response_queue))

    # Decoding
    decode_buffer = [[] for _ in range(request.num_samples)]
    parts = [[] for _ in range(request.num_samples)]

    def send_reset_buffer(sample_id):
        nonlocal decode_buffer
        if len(decode_buffer[sample_id]) == 0:
            return

        decoded = tokenizer.decode(decode_buffer[sample_id])
        part = ServeTextPart(text=decoded)

        if request.streaming:
            yield ServeStreamContentSequenceResponse(
                sample_id=sample_id, delta=ServeStreamContentSequenceDelta(part=part)
            )
        else:
            parts[sample_id].append(part)

        decode_buffer[sample_id] = []

    # Decode process
    finished = [False for _ in range(request.num_samples)]
    stats = {}
    idx = 0
    while True:
        response = response_queue.get()

        if response in ["stop", "error"]:
            break

        for sample_id, tokens in enumerate(response):
            if finished[sample_id]:
                continue

            if tokens[0] == im_end_id:
                finished[sample_id] = True
                if request.streaming:
                    yield from send_reset_buffer(sample_id)
                    yield ServeStreamContentSequenceResponse(
                        sample_id=sample_id,
                        finish_reason="stop",
                        stats=stats,
                    )
                continue

            is_semantic = (
                tokenizer.semantic_begin_id <= tokens[0] <= tokenizer.semantic_end_id
            )
            if is_semantic and request.streaming:
                yield from send_reset_buffer(sample_id)
                # Streaming vq
                _tokens = tokens[1:].clone()

                if config.share_codebook_embeddings is False:
                    for i in range(len(_tokens)):
                        _tokens[i] -= config.codebook_size * i

                yield ServeStreamContentSequenceResponse(
                    sample_id=sample_id,
                    delta=ServeStreamContentSequenceDelta(
                        part=ServeVQPart(codes=_tokens.tolist())
                    ),
                )
                continue

            # Not streaming vq
            if is_semantic:
                yield from send_reset_buffer(sample_id)
                # None streaming vq
                if len(parts[sample_id]) == 0 or not isinstance(
                    parts[sample_id][-1], ServeVQPart
                ):
                    _tokens = tokens[1:].clone()

                    if config.share_codebook_embeddings is False:
                        for i in range(len(_tokens)):
                            _tokens[i] -= config.codebook_size * i

                    parts[sample_id].append(ServeVQPart(codes=_tokens.tolist()))
                else:
                    for codebook_id, value in enumerate(tokens[1:, :]):
                        val = value.item()
                        if config.share_codebook_embeddings is False:
                            val -= config.codebook_size * codebook_id

                        parts[sample_id][-1].codes[codebook_id].append(val)
                continue

            if not is_semantic:
                # Stream text decode is not supported now
                decode_buffer[sample_id].append(tokens[0, 0])

        if idx == 0:
            stats["time_to_first_token"] = (time.time() - start) * 1000

        idx += 1

    for sample_id in range(request.num_samples):
        yield from send_reset_buffer(sample_id)

    stats["total_time"] = (time.time() - start) * 1000
    stats["total_tokens"] = idx

    if request.streaming:
        for sample_id in range(request.num_samples):
            if finished[sample_id]:
                continue
            yield ServeStreamContentSequenceResponse(
                finish_reason=response, stats=stats, sample_id=sample_id
            )
        return

    yield ServeContentSequenceResponse(
        content_sequences=[
            ServeContentSequenceParts(parts=parts[i])
            for i in range(request.num_samples)
        ],
        finish_reason=response,
        finished=finished,
        stats=stats,
    )


# For backward compatibility
execute_request = execute_chat_request


def execute_completion_request(
    input_queue: queue.Queue,
    tokenizer: FishTokenizer,
    request: ServeCompletionRequest,
    device="cuda:0",
):
    im_end_id = tokenizer.get_token_id(IM_END_TOKEN)
    prompt = torch.tensor(request.prompt, dtype=torch.long).to(device)

    req = {
        "prompt": prompt,
        "audio_masks": None,
        "audio_parts": None,
        "max_new_tokens": request.max_new_tokens,
        "im_end_id": im_end_id,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "repetition_penalty": request.repetition_penalty,
        "num_samples": request.num_samples,
        "early_stop_threshold": request.early_stop_threshold,
    }

    start = time.time()
    response_queue = queue.Queue()
    input_queue.put(GenerateRequest(req, response_queue))

    # Decode process
    finished = [False for _ in range(request.num_samples)]
    stats = {}
    idx = 0
    results = [[] for _ in range(request.num_samples)]

    while True:
        response = response_queue.get()

        if response in ["stop", "error"]:
            break

        for sample_id, tokens in enumerate(response):
            if finished[sample_id]:
                continue

            results[sample_id].append(tokens)

            if tokens[0] == im_end_id:
                finished[sample_id] = True
                continue

        if idx == 0:
            stats["time_to_first_token"] = (time.time() - start) * 1000

        idx += 1

    stats["total_time"] = (time.time() - start) * 1000
    stats["total_tokens"] = idx

    return ServeCompletionResponse(
        results=results,
        finish_reason=response,
        stats=stats,
    )


def swap_model(input_queue: queue.Queue, state_dict: dict):
    response_queue = queue.Queue()
    input_queue.put(SwapModelRequest(state_dict, response_queue))
    return response_queue.get()

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    from fish_speech.models.text2semantic.llama import (
        BaseTransformer,
        NaiveTransformer,
        DualARTransformer
    )
    from fish_speech.tokenizer import FishTokenizer
    from fish_speech.conversation import Conversation, Message, TextPart, VQPart
    
    def debug_conversation_encoding(conversation, tokenizer, config):
        """调试对话编码过程"""
        print("\n=== 调试对话编码 ===")
        
        # 可视化对话
        print("对话可视化:")
        conversation.visualize(tokenizer)
        
        # 编码对话
        prompt, audio_masks, audio_parts = conversation.encode_for_inference(
            tokenizer=tokenizer,
            num_codebooks=config.num_codebooks
        )
        
        print(f"Prompt形状: {prompt.shape}")
        print(f"第一个codebook的tokens: {prompt[0].tolist()}")
        
        # 解码最后几个token来查看
        last_tokens = prompt[0, -10:].tolist()
        print(f"最后10个tokens: {last_tokens}")
        for i, token_id in enumerate(last_tokens):
            try:
                decoded = tokenizer.decode([token_id])
                print(f"  Token {i}: {token_id} -> '{decoded}'")
            except:
                print(f"  Token {i}: {token_id} -> [无法解码]")
        
        return prompt, audio_masks, audio_parts
    
    def create_speech_conversation(system_prompt: str, user_text: str) -> Conversation:
        """创建用于语音生成的标准对话格式"""
        messages = [
            Message(
                role="system",
                parts=[TextPart(text=system_prompt)],
                modality="text"
            ),
            Message(
                role="user",
                parts=[TextPart(text=user_text)],
                modality="text"
            ),
            Message(
                role="assistant",
                parts=[],
                add_im_end=False,
                modality="voice"
            )
        ]
        return Conversation(messages=messages)
    
    def test_multiple_scenarios():
        """测试多种场景的VQ生成"""
        
        # 设置参数
        checkpoint_path = "checkpoints/openaudio-s1-mini"  # 请根据实际路径修改
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = torch.bfloat16
        
        print(f"使用设备: {device}")
        print(f"精度: {precision}")
        
        try:
            # 加载tokenizer和配置
            print("正在加载tokenizer...")
            tokenizer = FishTokenizer.from_pretrained(checkpoint_path)
            config = BaseModelArgs.from_pretrained(checkpoint_path)
            print(f"Tokenizer加载成功，词汇表大小: {tokenizer.vocab_size}")
            print(f"语义token范围: [{tokenizer.semantic_begin_id}, {tokenizer.semantic_end_id}]")
            
            # 加载模型
            print("正在加载模型...")
            model, decode_one_token = load_model(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=False
            )
            print(f"模型加载成功，类型: {type(model).__name__}")
            
            # 测试场景1: 遵循模板的基础对话
            print("\n=== 测试场景1: 标准模板对话 ===")
            conversation1 = create_speech_conversation(
                "Speak out the provided text.",
                "hello world"
            )
            test_generation_scenario("标准模板对话", conversation1, tokenizer, config, model, decode_one_token, device, precision)
            
            # 测试场景2: 中文指令
            print("\n=== 测试场景2: 中文指令 ===")
            conversation2 = create_speech_conversation(
                "请将提供的文本用语音读出来。",
                "你好世界"
            )
            test_generation_scenario("中文指令", conversation2, tokenizer, config, model, decode_one_token, device, precision)
            
            # 测试场景3: 更长的文本
            print("\n=== 测试场景3: 更长的文本 ===")
            conversation3 = create_speech_conversation(
                "Speak out the provided text.",
                "Hello, this is a test of the fish speech model."
            )
            test_generation_scenario("更长的文本", conversation3, tokenizer, config, model, decode_one_token, device, precision)
            
            # 测试场景4: 对话式指令
            print("\n=== 测试场景4: 对话式指令 ===")
            conversation4 = create_speech_conversation(
                "You are a helpful voice assistant. Respond to the user's request with speech.",
                "请说你好"
            )
            test_generation_scenario("对话式指令", conversation4, tokenizer, config, model, decode_one_token, device, precision)
            
        except Exception as e:
            print(f"测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_generation_scenario(scenario_name, conversation, tokenizer, config, model, decode_one_token, device, precision):
        """测试单个生成场景"""
        print(f"\n--- {scenario_name} ---")
        
        # 调试对话编码
        prompt, audio_masks, audio_parts = debug_conversation_encoding(conversation, tokenizer, config)
        
        prompt = prompt.to(device)
        audio_masks = audio_masks.to(device) if audio_masks is not None else None
        audio_parts = audio_parts.to(device, dtype=precision) if audio_parts is not None else None
        
        # 设置生成参数 - 调整参数来鼓励生成
        im_end_id = tokenizer.get_token_id(IM_END_TOKEN)
        generation_kwargs = {
            "prompt": prompt,
            "audio_masks": audio_masks,
            "audio_parts": audio_parts,
            "max_new_tokens": 50,  # 适合语音生成的token数量
            "im_end_id": im_end_id,
            "temperature": 0.8,  # 适中的温度设置
            "top_p": 0.95,
            "repetition_penalty": 1.1,  # 适度的重复惩罚
            "num_samples": 1,
            "early_stop_threshold": 0.6,  # 恢复早停机制
        }
        
        print(f"生成参数: temperature={generation_kwargs['temperature']}, top_p={generation_kwargs['top_p']}")
        
        # 生成tokens
        generated_tokens = []
        vq_tokens_found = False
        text_tokens = []
        vq_sequences = []
        
        print("开始生成...")
        for i, tokens in enumerate(generate(
            model=model,
            decode_one_token=decode_one_token,
            **generation_kwargs
        )):
            generated_tokens.append(tokens)
            
            # 检查是否包含语义token
            batch_tokens = tokens[0]  # 取第一个样本
            token_id = batch_tokens[0, 0].item()  # 第一个codebook的token
            
            is_semantic = (
                tokenizer.semantic_begin_id <= token_id <= tokenizer.semantic_end_id
            )
            
            if is_semantic:
                vq_tokens_found = True
                vq_codes = batch_tokens[1:].clone()  # 获取VQ codes（除了第一个codebook）
                vq_sequences.append(vq_codes)
                print(f"步骤 {i+1}: ✅ 发现VQ codes - Token ID: {token_id}, VQ形状: {vq_codes.shape}")
            elif token_id == im_end_id:
                print(f"步骤 {i+1}: 遇到结束token")
                break
            else:
                text_tokens.append(token_id)
                try:
                    decoded_text = tokenizer.decode([token_id])
                    print(f"步骤 {i+1}: 文本token - '{decoded_text}' (ID: {token_id})")
                except:
                    print(f"步骤 {i+1}: 未知token - ID: {token_id}")
            
            # 增加显示长度
            if i >= 20:  # 适当的显示步数限制
                print("达到最大显示步数，停止...")
                break
        
        # 结果分析
        print(f"\n{scenario_name} 结果:")
        if vq_tokens_found:
            print(f"✅ 成功生成VQ codes！数量: {len(vq_sequences)}")
            if vq_sequences:
                vq_tensor = torch.stack(vq_sequences, dim=-1)
                print(f"VQ codes形状: {vq_tensor.shape}, 范围: [{vq_tensor.min().item()}, {vq_tensor.max().item()}]")
        else:
            print(f"❌ 未生成VQ codes")
        
        if text_tokens:
            try:
                decoded_text = tokenizer.decode(text_tokens)
                print(f"生成的文本: '{decoded_text}'")
            except:
                print(f"文本token数量: {len(text_tokens)}")
        
        return vq_tokens_found
    
    def main():
        """主函数"""
        parser = argparse.ArgumentParser(description="测试Fish Speech模型VQ codes生成")
        parser.add_argument(
            "--checkpoint", 
            type=str, 
            default="checkpoints/openaudio-s1-mini",
            help="模型checkpoint路径"
        )
        
        args = parser.parse_args()
        
        print("=== Fish Speech VQ Generation 多场景测试 ===")
        print(f"检查点路径: {args.checkpoint}")
        
        if not Path(args.checkpoint).exists():
            print(f"错误: 检查点路径不存在: {args.checkpoint}")
            print("请确保路径正确，或者使用 --checkpoint 参数指定正确的路径")
            return
        
        success = test_multiple_scenarios()
        
        if success:
            print("\n🎉 测试成功！模型能够正常生成VQ codes")
        else:
            print("\n❌ 测试失败！请检查模型和配置")
        
        print("\n=== 调试建议 ===")
        print("1. 确保使用正确的系统提示词引导模型生成语音")
        print("2. 检查模型是否正确加载了语音生成权重")
        print("3. 尝试调整temperature (0.7-1.0) 和top_p (0.9-0.95) 参数")
        print("4. 确认tokenizer的语义token范围配置正确")
        print("5. 检查对话格式是否符合模型训练时的格式")
        print("6. 如果仍有问题，可以尝试不同的系统提示词")
        print("\n=== 推荐的系统提示词 ===")
        print("英文: 'Speak out the provided text.'")
        print("中文: '请将提供的文本用语音读出来。'")
        print("对话式: 'You are a helpful voice assistant. Respond with speech.'")
    
    main()
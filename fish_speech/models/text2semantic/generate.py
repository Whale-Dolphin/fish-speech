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


from flash_fish_inference.models.llama import (
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
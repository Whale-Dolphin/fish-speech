from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
import os
import queue
import threading
import time
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Callable

import click
import numpy as np
import torch
import torch._inductor.config
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech.conversation import (
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from fish_speech.text import clean_text, split_text
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True


def multinomial_sample_one_no_sync(
    probs_sort,
):
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
    # print(f"logits shape: {logits.shape}, previous_tokens shape: {previous_tokens.shape if previous_tokens is not None else None}")
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[:, -1], previous_tokens=previous_tokens, **sampling_kwargs
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
    """
    Generate one token using dual autoregressive transformer for text-to-speech.
    
    First generates semantic tokens, then generates acoustic codebook tokens sequentially.
    
    Args:
        x: Input token tensor (1, num_codebooks+1, seq_len)
        input_pos: Position indices for input tokens (seq_len,)
        temperature/top_p/repetition_penalty: Sampling parameters (1, 1)
        previous_tokens: Previous tokens for repetition penalty (1, num_codebooks+1, history_seq_len)
        audio_masks/audio_parts: Audio conditioning tensors (num_codebooks, seq_len)
    
    Returns:
        Generated tokens tensor (num_codebooks+1, 1) - one token per codebook
    """
    # print(x, torch.count_nonzero(vq_masks))
    x = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]
    # print(f"logits shape: {logits.shape}, hidden_states shape: {hidden_states.shape}")

    codebooks = [
        sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[:, 0] if previous_tokens is not None else None
            ),
        )[0]
    ]
    # print(f"text_codebooks:{codebooks}")

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor(
        [0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    # print(f"codebooks[0]:{codebooks[0]}")
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits[:, :, :4096]

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

    return codebooks


def decode_n_tokens(
    model: Callable,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    decode_one_token=decode_one_token_ar,
) -> torch.Tensor:
    """
    Generate n tokens iteratively using the model.
    
    Args:
        model: The transformer model
        cur_token: Current token tensor of shape (1, num_codebooks+1, seq_len)
        input_pos: Current input position tensor
        num_new_tokens: Number of new tokens to generate
        semantic_ids: List of semantic token IDs
        decode_one_token: Function to decode one token
        **sampling_kwargs: Additional sampling parameters
    
    Returns:
        Generated tokens tensor of shape (num_codebooks+1, generated_len)
    """
    previous_tokens = torch.zeros(
        (1, model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    start_time = time.time()

    for i in tqdm(range(num_new_tokens)):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :, :win_size]
        else:
            window = previous_tokens[:, :, i - win_size: i]

        with sdpa_kernel(
            SDPBackend.MATH
        ):
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
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, :, i: i + 1] = next_token.view(
            1, model.config.num_codebooks + 1, -1
        )

        # print(f"cur_token: {cur_token}")
        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break
    end_time = time.time()
    print(f"Decoding {i} tokens took {end_time - start_time:.2f} seconds")

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: BaseTransformer,
    prompt: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    max_new_tokens: int,
    decode_one_token: Callable,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Generate audio tokens from text prompt using the transformer model.
    
    Args:
        model: The transformer model for generation
        prompt: Input token tensor of shape (num_codebooks+1, seq_len)
        audio_masks: Audio mask tensor for conditioning (num_codebooks, seq_len)
        audio_parts: Audio part tensor for conditioning (num_codebooks, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        decode_one_token: Function to decode one token at a time
        **sampling_kwargs: Additional sampling parameters (temperature, top_p, repetition_penalty)
    
    Returns:
        Generated sequence tensor of shape (1, num_codebooks+1, total_seq_len)
        where total_seq_len = original_seq_len + generated_tokens_len
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    logger.debug(f"Input prompt shape: {prompt.shape}, T: {T}")
    # Ensure prompt is of shape (1, num_codebooks+1, seq_len)
    prompt = prompt.unsqueeze(0)

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
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    codebook_dim = 1 + model.config.num_codebooks
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    temperature = torch.tensor(
        [[sampling_kwargs["temperature"]]], device=device, dtype=torch.bfloat16
    )
    top_p = torch.tensor(
        [[sampling_kwargs["top_p"]]], device=device, dtype=torch.bfloat16
    )
    repetition_penalty = torch.tensor(
        [[sampling_kwargs["repetition_penalty"]]
         ], device=device, dtype=torch.bfloat16
    )

    first_token = (
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
        .view(1, codebook_dim, -1)
    )
    logger.debug(f"First token shape: {first_token}")
    seq[:, T: T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        first_token,
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        decode_one_token=decode_one_token,
    )

    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1:] = x

    return seq


def load_model(checkpoint_path, device, precision):
    model: DualARTransformer = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True
    )

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    return model.eval()


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def check_prompt_tokens(prompt_tokens: torch.Tensor, num_codebooks: int):
    """
    Check if the prompt tokens are valid for the given number of codebooks.
    
    Args:
        prompt_tokens (torch.Tensor): The prompt tokens tensor.
        num_codebooks (int): The number of codebooks expected.
    
    Returns:
        torch.Tensor: The checked and possibly modified prompt tokens tensor.
    """
    if prompt_tokens.ndim == 3:
        assert (
            prompt_tokens.shape[0] == 1
        ), "3D prompt tokens should have shape (1, num_codebooks, seq_len)"
        prompt_tokens = prompt_tokens[0]

    assert prompt_tokens.ndim == 2, "Prompt tokens should be 2D tensor"

    if prompt_tokens.shape[0] > num_codebooks:
        logger.warning(
            f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
        )
        prompt_tokens = prompt_tokens[:num_codebooks]

    return prompt_tokens


def generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: Callable,
    text: str,
    max_new_tokens: int = 0,
    top_p: float = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    iterative_prompt: bool = True,
    chunk_length: int = 150,
    prompt_text: Optional[str] = None,
    prompt_tokens: Optional[torch.Tensor] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens), "Prompt text and tokens must have the same length"

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer

    texts = split_text(text, chunk_length) if iterative_prompt else [text]

    messages = [
        Message(
            role="system",
            parts=[TextPart(text="Speak out the provided text.")],
            cal_loss=False,
        )
    ]

    for text in texts:
        logger.debug(f"Processing text chunk: {text}")
        messages.append(
            Message(
                role="user",
                parts=[TextPart(text=clean_text(text))],
                cal_loss=False,
            )
        )

        logger.info(f"Encoded text: {text}")

    logger.debug(f"prompt_text: {prompt_text}, {type(prompt_text)}")
    if use_prompt:
        messages.append(
            Message(
                role="user",
                parts=[TextPart(text=clean_text(prompt_text))],
                cal_loss=False,
            )
        )
        check_prompt_tokens(prompt_tokens, model.config.num_codebooks)

        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>"), VQPart(
                    codes=prompt_tokens.to(device))],
                cal_loss=False,
            )
        )
    else:
        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>")],
                cal_loss=False,
                add_im_end=False,
            )
        )

    logger.debug(f"Messages: {messages}")
    print(f"Messages: {messages}")

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    conversation = Conversation(messages=messages)
    encoded, audio_masks, audio_parts = conversation.encode_for_inference(
        tokenizer=tokenizer,
        num_codebooks=model.config.num_codebooks,
    )

    encoded = encoded.to(device)
    audio_masks = audio_masks.to(device) if audio_masks is not None else None
    audio_parts = (
        audio_parts.to(device, dtype=torch.bfloat16)
        if audio_parts is not None
        else None
    )

    t0 = time.perf_counter()
    y = generate(
        model=model,
        prompt=encoded,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        max_new_tokens=max_new_tokens,
        decode_one_token=decode_one_token,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # if compile:
    #     logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t = time.perf_counter() - t0

    prompt_length = encoded.size(1)
    tokens_generated = y.size(1) - prompt_length
    print(f"Generated {tokens_generated} tokens in {t:.02f} seconds")
    tokens_sec = tokens_generated / t
    logger.info(
        f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec")
    logger.info(
        f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

    if torch.cuda.is_available():
        logger.info(
            f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    # Put the generated tokens
    # since there is <im_end>, we remove last token
    codes = y[1:, prompt_length + 1:].clone()
    assert (codes >= 0).all(), f"Negative code found"

    return codes

    # decoded = y[:, prompt_length:].clone()
    # # But for global encoding, we should keep the <im_end> token

    # global_encoded.append(decoded)
    # assert (codes >= 0).all(), f"Negative code found: {codes}"
    # ield GenerateResponse(action="sample", codes=codes, text=texts[seg_idx])
    # seg_idx += 1

    # yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = load_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(
                            status="success", response=chunk)
                    )
            except Exception as e:
                response_queue.put(WrappedGenerateResponse(
                    status="error", response=e))

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


@click.command()
@click.option(
    "--text",
    type=str,
    default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None)
@click.option(
    "--prompt-tokens-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.7)
@click.option("--repetition-penalty", type=float, default=1.2)
@click.option("--temperature", type=float, default=0.7)
@click.option("--checkpoint-path", type=str, default="checkpoints/openaudio-s1-mini")
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=100)
@click.option("--output-dir", type=Path, default="temp")
def main(
    text: str,
    prompt_text: Optional[str],
    prompt_tokens_path: Optional[Path],
    max_new_tokens: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    logger.info("Loading model ...")
    t0 = time.time()

    model = load_model(checkpoint_path, device, precision)
    model_type_str = model.__class__.__name__

    if model_type_str == "DualARTransformer":
        decode_one_token = decode_one_token_ar
    else:
        raise ValueError(
            f"Unsupported model type: {model_type_str}. Supported types are: DualARTransformer."
        )

    if compile:
        decode_one_token = torch.compile(
            decode_one_token,
            fullgraph=True,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
        )

    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    if prompt_tokens_path is not None:
        prompt_tokens = torch.from_numpy(np.load(prompt_tokens_path)).to(device)
    else:
        prompt_tokens = None

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

    logger.debug(f"Generator type: {generator}")
    sys.exit(0)

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, torch.cat(codes, dim=1).cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")
            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()

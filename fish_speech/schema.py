import os
from dataclasses import Field
from typing import Annotated, Literal, Optional

import torch
from pydantic import AfterValidator, BaseModel, confloat, conint, conlist
from pydantic.functional_validators import SkipValidation

from fish_speech.conversation import AudioPart as ConvAudioPart
from fish_speech.conversation import Message
from fish_speech.conversation import TextPart as ConvTextPart
from fish_speech.conversation import VQPart as ConvVQPart

GLOBAL_NUM_SAMPLES = int(os.getenv("GLOBAL_NUM_SAMPLES", 1))
IS_SGLANG_BACKEND = os.getenv("SGLANG_BACKEND", "false").lower() == "true"


class ServeVQPart(BaseModel):
    type: Literal["vq"] = "vq"
    codes: SkipValidation[list[list[int]]]


class ServeTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ServeAudioPart(BaseModel):
    type: Literal["audio"] = "audio"
    features: list[list[float]]


class ServeMessage(BaseModel):
    role: Literal["system", "assistant", "user", "raw"]
    parts: list[ServeVQPart | ServeTextPart | ServeAudioPart]
    modality: Literal["text", "voice", "interleave"] | None = None

    def to_conversation_message(self):
        new_message = Message(
            role=self.role,
            parts=[],
            modality=self.modality,
        )

        for part in self.parts:
            if isinstance(part, ServeTextPart):
                new_message.parts.append(ConvTextPart(text=part.text))
            elif isinstance(part, ServeVQPart):
                new_message.parts.append(
                    ConvVQPart(codes=torch.tensor(part.codes, dtype=torch.int))
                )
            elif isinstance(part, ServeAudioPart):
                new_message.parts.append(
                    ConvAudioPart(
                        features=torch.tensor(part.features, dtype=torch.float32)
                    )
                )
            else:
                raise ValueError(f"Unsupported part type: {part}")

        return new_message


class ServeChatRequest(BaseModel):
    messages: Annotated[list[ServeMessage], conlist(ServeMessage, min_length=1)]
    max_new_tokens: int = 600
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    streaming: bool = False
    num_samples: int = 1
    early_stop_threshold: float = 1.0
    modality: Literal["text", "voice", "interleave"] = "voice"


class ServeContentSequenceRequest(BaseModel):
    # Raw content sequence dict that we can use with ContentSequence(**content)
    content: dict
    max_new_tokens: int = 600
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    streaming: bool = False
    num_samples: int = 1
    early_stop_threshold: float = 1.0


class ServeCompletionRequest(BaseModel):
    prompt: list[list[int]]
    max_new_tokens: int = 600
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    num_samples: int = 1
    early_stop_threshold: float = 1.0


class ServeChatResponse(BaseModel):
    messages: list[ServeMessage]
    finish_reason: Literal["stop", "error"] | None = None
    stats: dict[str, int | float | str] = {}


class ServeContentSequenceParts(BaseModel):
    parts: list[ServeVQPart | ServeTextPart | ServeAudioPart]


class ServeContentSequenceResponse(BaseModel):
    content_sequences: list[ServeContentSequenceParts]
    finish_reason: Literal["stop", "error"] | None = None
    stats: dict[str, int | float | str] = {}
    finished: list[bool] | None = None


class ServeCompletionResponse(BaseModel):
    results: list[list[int]]
    finish_reason: Literal["stop", "error"] | None = None
    stats: dict[str, int | float | str] = {}


class ServeSwapModelRequest(BaseModel):
    state_dict: bytes


class ServeStreamDelta(BaseModel):
    role: Literal["system", "assistant", "user"] | None = None
    part: ServeVQPart | ServeTextPart | None = None


class ServeStreamResponse(BaseModel):
    sample_id: int = 0
    delta: ServeStreamDelta | None = None
    finish_reason: Literal["stop", "error"] | None = None
    stats: dict[str, int | float | str] | None = None


class ServeStreamContentSequenceDelta(BaseModel):
    part: ServeVQPart | ServeTextPart | None = None


class ServeStreamContentSequenceResponse(BaseModel):
    sample_id: int = 0
    delta: ServeStreamContentSequenceDelta | None = None
    finish_reason: Literal["stop", "error"] | None = None
    stats: dict[str, int | float | str] | None = None


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


def allowed_num_samples(v: int) -> int:
    if IS_SGLANG_BACKEND:
        assert 1 <= v <= 8
    else:
        assert v in [1, GLOBAL_NUM_SAMPLES]

    return v


class ServeProsody(BaseModel):
    speed: Annotated[float, confloat(ge=0.5, le=2.0)] = 1.0
    volume: Annotated[float, confloat(ge=-10, le=10)] = 0.0


class ServeTTSRequest(BaseModel):
    text: str

    # Each chunk has a maximum of 1024 tokens (50s) by default
    max_new_tokens: int = 600
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    min_chunk_length: Annotated[int, conint(ge=0, le=100, strict=True)] = 50
    format: Literal["wav", "pcm", "mp3", "opus"] = "mp3"
    sample_rate: int | None = None
    mp3_bitrate: Literal[64, 128, 192] = 128
    opus_bitrate: Literal[-1000, 24, 32, 48, 64] = 32  # -1000 means auto
    # Prosody
    prosody: ServeProsody | None = None

    # References audios
    references: list[ServeReferenceAudio] = []
    # Limit the number of history messages to 1. save compute 2. reduce accumulation of errors
    max_history_messages: Annotated[int, conint(ge=0, le=10, strict=True)] = 1
    # Reset history every n steps to reduce accumulation of errors
    reset_history_every_n_steps: Annotated[int, conint(ge=0, le=50, strict=True)] = 0

    # Normalize before TTS
    normalize: bool = True
    language: (
        Literal[
            "zh", "en", "ja", "de", "fr", "es", "ko", "ar", "ru", "nl", "it", "pl", "pt"
        ]
        | None
    ) = None

    # Early stop if fraction of samples are finished
    early_stop_threshold: float = 1

    # Rerank
    rerank_mode: Literal["reward", "asr"] = "reward"


class ServeTTSBatchRequest(ServeTTSRequest):
    num_samples: Annotated[int, AfterValidator(allowed_num_samples)] = 1


class ServeTTSResponse(BaseModel):
    audios: list[bytes]


class ServeInternalTTSCandidate(BaseModel):
    audio: bytes
    codes: bytes  # numpy bytes


class ServeInternalTTSResponse(BaseModel):
    candidates: list[ServeInternalTTSCandidate]


class ServeASRRequest(BaseModel):
    # The audio should be an uncompressed PCM float16 audio
    audios: list[bytes]
    sample_rate: int = 44100
    language: Literal["zh", "en", "ja", "auto"] = "auto"


class ServeASRTranscription(BaseModel):
    text: str
    duration: float
    huge_gap: bool


class ServeASRSegment(BaseModel):
    text: str
    start: float
    end: float


class ServeTimedASRResponse(BaseModel):
    text: str
    segments: list[ServeASRSegment]
    duration: float


class ServeASRResponse(BaseModel):
    transcriptions: list[ServeASRTranscription]


class ServeRewardRequest(BaseModel):
    messages: list[list[ServeMessage]]


class ServeRewardResponse(BaseModel):
    rewards: list[float]


class ServeVQGANEncodeRequest(BaseModel):
    # The audio here should be in wav, mp3, etc
    audios: list[bytes]


class ServeVQGANEncodeResponse(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeRequest(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeResponse(BaseModel):
    # The audio here should be in PCM float16 format
    audios: list[bytes]


class ServeStreamTTSRequest(ServeTTSRequest):
    text: str = ""
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 100


class ServeStreamTTSStartEvent(BaseModel):
    event: Literal["start"] = "start"
    debug: bool = False
    request: ServeStreamTTSRequest


class ServeStreamTTSAddTextEvent(BaseModel):
    event: Literal["text"] = "text"
    text: str


class ServeStreamTTSStopEvent(BaseModel):
    event: Literal["stop"] = "stop"


class ServeStreamTTSFlushEvent(BaseModel):
    event: Literal["flush"] = "flush"


class ServeStreamTTSLogEvent(BaseModel):
    event: Literal["log"] = "log"
    message: str
    time: float | None = None


class ServeStreamTTSFinishEvent(BaseModel):
    event: Literal["finish"] = "finish"
    reason: Literal["stop", "error"]
    time: float | None = None
    message: str | None = None


class ServeStreamTTSAudioEvent(BaseModel):
    event: Literal["audio"] = "audio"
    audio: bytes
    time: float | None = None

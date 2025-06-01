import struct
from dataclasses import asdict
from typing import Any, Iterable, Type, TypeVar

import numpy as np
import ormsgpack
import torch
from pydantic import BaseModel


def default_handler(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()

    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": obj.shape,
            "dtype": obj.dtype.str,
            "data": obj.tobytes(),
        }

    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def restore_ndarray(obj, to_tensor: bool = False):
    if isinstance(obj, dict) and "__ndarray__" in obj:
        obj = np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])

    if to_tensor and isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj.copy())

    return obj


def pack_mpack_item(data: Any) -> bytes:
    if isinstance(data, dict):
        data_dict = data
    elif isinstance(data, BaseModel):
        data_dict = data.model_dump()
    else:
        data_dict = asdict(
            data
        )  # Attempt to convert, may raise error if not compatible

    return ormsgpack.packb(data_dict, default=default_handler)


def pack_mpack_stream_item(data: Any) -> bytes:
    buf = pack_mpack_item(data)
    return struct.pack("I", len(buf)) + buf


T = TypeVar("T")


def read_mpack_stream(f, type: Type[T]) -> Iterable[T]:
    while True:
        buf = f.read(4)
        if len(buf) == 0:
            break
        size = struct.unpack("I", buf)[0]
        buf = f.read(size)
        text_data = type(**ormsgpack.unpackb(buf))
        yield text_data

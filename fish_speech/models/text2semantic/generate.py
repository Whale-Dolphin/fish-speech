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
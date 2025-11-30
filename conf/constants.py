# Copyright (c) 2025 Huawei. All rights reserved.

from enum import Enum


class Stage(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


SIZE_OF_FP16 = 2
GB = 1e9
TB = 1e12
BILLION = 1e9
SIZE_OF_W = 2
SIZE_OF_A = 2
SIZE_OF_KV = 2
MILISECOND_FACTOR = 1000
UI_FACTOR = 100
VLLM_URL = 'http://10.174.98.151:8010'
ORCHESTRATOR_URL = 'http://10.174.98.151:4004'
PREDSIM_DEFAULT_PORT = 8081
MODEL_ID = "blue"
TOKENIZER_PATH = "/mnt/disk2/llama_models/llama7B/llama2-7b-tokenizer"


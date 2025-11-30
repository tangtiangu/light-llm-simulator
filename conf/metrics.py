# Copyright (c) 2025 Huawei. All rights reserved.

from enum import Enum


class KPIs(Enum):
    TTFT = "TTFT"
    TBT = "TBT"
    PREFILL_MEMORY = "prefill_memory"
    DECODE_MEMORY = "decode_memory"
    THROUGHPUT_PREFILL = "throughput_prefill"
    THROUGHPUT_DECODE = "throughput_decode"

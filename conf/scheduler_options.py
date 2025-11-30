# Copyright (c) 2025 Huawei. All rights reserved.

from enum import Enum


class SchedulerStrategy(Enum):
    FCFS = "FCFS"
    PNUELI_LAB = "PNUELI_LAB"
    OUTPUT_LENGTH = "OUTPUT_LENGTH"
    PRIORITY = "PRIORITY"
    INPUT_LENGTH_SJF = "INPUT_LENGTH_SJF"
    INPUT_LENGTH_LJF = "INPUT_LENGTH_LJF"


class PolicyType(Enum):
    MINIMUM_LATENCY = "minimum_latency"
    MAXIMUM_THROUGHPUT = "maximum_throughput"
    MINIMUM_RESOURCES = "minimum_resources"

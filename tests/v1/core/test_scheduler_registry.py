# test_ewsjf_scheduler.py

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.core.sched import SCHEDULER_REGISTRY
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams

def test_ewsjf_scheduler_ordering():
    model_config = ModelConfig(model="facebook/opt-125m")
    vllm_config = VllmConfig(model_config=model_config)

    scheduler_cls = SCHEDULER_REGISTRY["ewsjf"]
    scheduler = scheduler_cls(vllm_config, vllm_config.cache_config)

    # Create synthetic requests with varying max_tokens
    requests = [
        Request("r1", [1, 2], None, None, None, SamplingParams(max_tokens=20), None, eos_token_id=2),
        Request("r2", [3, 4], None, None, None, SamplingParams(max_tokens=10), None, eos_token_id=2),
        Request("r3", [5, 6], None, None, None, SamplingParams(max_tokens=5), None, eos_token_id=2),
    ]

    for req in requests:
        scheduler.add_request(req)

    scheduled = scheduler.schedule()
    scheduled_ids = [r.request_id for r in scheduled]

    assert scheduled_ids == ["r3", "r2", "r1"], "EWSJFScheduler did not prioritize shortest jobs first"

# benchmark_schedulers.py

import time
from vllm.config import ModelConfig, VllmConfig
from vllm.v1.core.sched import SCHEDULER_REGISTRY
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams

def generate_requests(n):
    return [
        Request(
            request_id=f"r{i}",
            prompt_token_ids=[101, 102],
            multi_modal_kwargs=None,
            multi_modal_hashes=None,
            multi_modal_placeholders=None,
            sampling_params=SamplingParams(max_tokens=(i % 50) + 1),
            pooling_params=None,
            eos_token_id=2,
            arrival_time=i * 0.01,
            priority=i % 5,
        )
        for i in range(n)
    ]

def benchmark_scheduler(name, num_requests=1000):
    model_config = ModelConfig(model="facebook/opt-125m")
    vllm_config = VllmConfig(model_config=model_config)

    scheduler_cls = SCHEDULER_REGISTRY[name]
    scheduler = scheduler_cls(vllm_config, vllm_config.cache_config)

    requests = generate_requests(num_requests)
    for req in requests:
        scheduler.add_request(req)

    start = time.time()
    scheduled = scheduler.schedule()
    end = time.time()

    print(f"{name.upper():<10} | Scheduled {len(scheduled)} requests in {end - start:.4f} seconds")

if __name__ == "__main__":
    for name in ["fcfs", "priority", "ewsjf"]:
        benchmark_scheduler(name)

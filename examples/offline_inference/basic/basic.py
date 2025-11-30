import vllm.v1.worker.simulator_config
import vllm.v1.worker.mock_model_runner
from vllm import LLM, SamplingParams, AsyncEngineArgs, AsyncLLMEngine  
import vllm.envs as envs  
  
import asyncio
import os
import sys
import time
import random
import pandas as pd


from conf.hardware_config import HardwareTopology, DeviceType
from datasets import load_dataset
# os.environ["VLLM_USE_V1"] = "1"

# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

class VLLMRunner:
    def __init__(self, model_name: str, dataset_name: str, hw_topology: dict, dataset_subset: str, dataset_key:str,  dataset_split: str = "test"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_split = dataset_split
        self.dataset_key = dataset_key
        self.llm = None
        self.prompts = None
        self.hw_topology = hw_topology

    # def load_data(self, n_samples: int = 10):
    #     df = pd.read_csv(self.csv_path)
    #     input_list = df['input'].tolist()
    #     random.shuffle(input_list)
    #     self.prompts = input_list[:n_samples]
    def load_data(self, n_samples: int = 10, text_column: str = "train"):
        """Load data from Hugging Face dataset"""
        if self.dataset_subset:
            dataset = load_dataset(self.dataset_name, self.dataset_subset)
        else:
            dataset = load_dataset(self.dataset_name)
        
        # Extract text from the specified column
        if text_column in dataset.column_names:
            input_list = dataset[text_column]
        else:
            raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {dataset.column_names}")
        
        # random.shuffle(input_list)
        self.prompts = input_list[self.dataset_key][:n_samples]


    def init_model(self):        

            self.llm = LLM(model=self.model_name, pipeline_parallel_size= 1,
                        tensor_parallel_size= 1,
                        enforce_eager=True, 
                        distributed_executor_backend="mp",
                       device_name_config=self.hw_topology["type"],
                    number_of_ranks=self.hw_topology["number_of_ranks"],
                       npus_per_rank=self.hw_topology["npus_per_rank"],)

    def run_vllm(self, temperature = 0.8, top_p = 0.95, top_k = 0.95, min_tokens = 2, max_tokens = 15):
        if self.llm is None:
            self.init_model()
        if self.prompts is None:
            self.load_data()

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k,
                                         min_tokens=min_tokens, max_tokens=max_tokens, logprobs=None)

        start = time.time()
        outputs = self.llm.generate(self.prompts, sampling_params)
        end = time.time()

        results = []
        for output in outputs:
            results.append({
                "prompt": output.prompt,
                "output": output.outputs[0].text
            })

        metrics = self.metrics(end - start, self.prompts, outputs)

        return metrics

    async def generate_async(self, engine, request_id, prompt, sampling_params):
        results_generator = engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output

    async def send_requests_with_rate_limit(self, engine, prompts, sampling_params, requests_per_second=15):
        tasks = []
        total = len(prompts)
        interval = 1.0 / requests_per_second

        for i, prompt in enumerate(prompts):
            request_id = f"request_{i}"

            task = asyncio.create_task(
                self.generate_async(engine, request_id, prompt, sampling_params)
            )
            tasks.append(task)

            sent_percent = int(((i + 1) / total) * 100)
            print(f"Adding requests: {sent_percent}%")

            if i < total - 1:
                await asyncio.sleep(interval)

        return tasks

    async def run_async_vllm(self, temperature=0.8, top_p=0.95, top_k=0.95, min_tokens=10, max_tokens=15, rate=15):
        if self.prompts is None:
            self.load_data()

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            logprobs=None
        )
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            enforce_eager=True,  
            pipeline_parallel_size= 2,
            distributed_executor_backend="mp",
            device_name_config=self.hw_topology
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)

        total = len(self.prompts)
        start = time.time()

        tasks = await self.send_requests_with_rate_limit(llm, self.prompts, sampling_params, requests_per_second=rate)

        completed = 0
        outputs = []
        for task in asyncio.as_completed(tasks):
            result = await task
            outputs.append(result)
            completed += 1
            percent = int((completed / total) * 100)
            print(f"Processed prompts: {percent}%")

        end = time.time()
        duration = end - start

        return self.metrics(duration, self.prompts, outputs)

    def metrics(self, total_runtime, prompts, outputs):
        """
        Calculate aggregate metrics without printing them.
        Returns a dictionary with results.
        """
        num_requests = len(prompts)
        total_generated_tokens = 0
        total_prompt_tokens = 0

        for output in outputs:
            prompt_len = len(output.prompt_token_ids)
            generated_len = len(output.outputs[0].token_ids)
            total_prompt_tokens += prompt_len
            total_generated_tokens += generated_len

        if total_runtime > 0:
            requests_per_second = num_requests / total_runtime
            output_tokens_per_second = total_generated_tokens / total_runtime
        else:
            requests_per_second = 0
            output_tokens_per_second = 0

        return {
            "total_runtime_sec": total_runtime,
            "num_requests": num_requests,
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "requests_per_second": requests_per_second,
            "output_tokens_per_second": output_tokens_per_second,
            "note": "TTFT and per-request latencies are not available in this version."
        }


if __name__ == "__main__":
    hw_topology = {"number_of_ranks":1,
                  "npus_per_rank":8,
                  "type":DeviceType.NVidiaA100.value}


    vllm_runner = VLLMRunner(
            model_name="NousResearch/Hermes-3-Llama-3.1-8B",
            dataset_name="nvidia/Nemotron-RL-math-OpenMathReasoning", #"Amod/mental_health_counseling_conversations",#'yehzw/wikitext-103',
            dataset_subset=None,
            dataset_key="question",
            dataset_split='train',
            hw_topology=hw_topology)

    res = vllm_runner.run_vllm(temperature=0.0, top_p=1.0,top_k=1, min_tokens=2, max_tokens=15)

    print(res)
    # res = asyncio.run(vllm_runner.run_async_vllm(temperature=0.8, top_p=0.95, min_tokens=10, max_tokens=15, rate=50))




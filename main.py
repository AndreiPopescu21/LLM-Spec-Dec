import sys
import os

# Get the absolute path of the local vllm repo
repo_path = os.path.abspath("vllm")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Insert at the beginning of sys.path to prioritize it
sys.path.insert(0, repo_path)


from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams
import os

# os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

class BaseLLMPipeline(ABC):
    def __init__(self, sampling_params: SamplingParams, **llm_kwargs):
        self.sampling_params = sampling_params
        self.llm_kwargs = llm_kwargs  # Custom parameters for LLM
        self.llm = None  # to be defined in subclasses

    @abstractmethod
    def setup_model(self):
        """Set up the LLM based on the decoding method."""
        pass

    def start_profile(self):
      if not self.llm:
            self.setup_model()
      self.llm.start_profile()

    def stop_profile(self):
      if not self.llm:
            self.setup_model()
      self.llm.stop_profile()

    def generate(self, prompts: list):
        if not self.llm:
            self.setup_model()
        return self.llm.generate(prompts, self.sampling_params)

# pipeline/naive_pipeline.py
# from pipeline.base_pipeline import BaseLLMPipeline
from vllm import LLM, SamplingParams
import os
from collections import namedtuple

Output = namedtuple("Output", ["outputs"])
SingleOutput = namedtuple("SingleOutput", ["token_ids", "text"])

from vllm import LLM, SamplingParams

class NaiveLLMPipeline(BaseLLMPipeline):
    def setup_model(self):
        default_params = {"model": "facebook/opt-6.7b"}
        # default_params = {"model": "facebook/opt-125m"}
        # Merge custom params with defaults; custom params can override defaults
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

class DraftModelLLMPipeline(BaseLLMPipeline):
    def setup_model(self):
        default_params = {
            "model": "facebook/opt-6.7b",
            "tensor_parallel_size": 1,
            "speculative_model": "facebook/opt-125m",
            "num_speculative_tokens": 5,
            # "spec_decoding_acceptance_method": "top_p_speculative_sampler",
        }
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

class NGramLLMPipeline(BaseLLMPipeline):
    def setup_model(self):
        default_params = {
            "model": "facebook/opt-6.7b",
            "tensor_parallel_size": 1,
            "speculative_model": "[ngram]",
            "num_speculative_tokens": 3,
            "ngram_prompt_lookup_max": 3,
        }
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

class MLPSpecLLMPipeline(BaseLLMPipeline):
    def setup_model(self):
        default_params = {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "tensor_parallel_size": 1,
            "speculative_model": "ibm-ai-platform/llama3-70b-accelerator",
            "speculative_draft_tensor_parallel_size": 1,
        }
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

# benchmark/benchmark.py
import time, json
from tqdm import tqdm
import numpy as np

def benchmark_pipeline(pipeline, prompts: list):
    pipeline.generate(prompts)
    pipeline.generate(prompts)
    start = time.time()
    outputs = pipeline.generate(prompts)
    end = time.time()
    print((end - start) / sum([len(o.outputs[0].token_ids) for o in outputs]))
    # Print the outputs.
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"text: {generated_text!r}")

    return outputs

# TODO: set np seed
class LatencyBenchmark():
    def __init__(self, num_iters=10, num_iters_warmup=3, output_json=None, batch_size=4, input_len=20):
        self.num_iters = num_iters
        self.num_iters_warmup = num_iters_warmup
        self.output_json = output_json
        self.batch_size = batch_size
        self.input_len = input_len
        self.prompts = self.generate_random_prompts()
        self.warmup_prompts = self.generate_random_prompts()

    def generate_random_prompts(self):
        dummy_prompt_token_ids = np.random.randint(32, size=(self.batch_size, self.input_len))
        return [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]

    def run_to_completion(self, pipeline, prompts):
        """Measures latency of the pipeline"""
        start_time = time.perf_counter()
        pipeline.generate(prompts)
        end_time = time.perf_counter()
        return end_time - start_time

    def benchmark(self, pipeline):
        """Runs the benchmark and prints latency statistics"""
        print("Warming up...")
        for _ in tqdm(range(self.num_iters_warmup), desc="Warmup iterations"):
            self.run_to_completion(pipeline, self.warmup_prompts)

        # Benchmark.
        latencies = []
        for _ in tqdm(range(self.num_iters), desc="Profiling iterations"):
            latencies.append(self.run_to_completion(pipeline, self.prompts))
        latencies = np.array(latencies)
        
        percentages = [10, 25, 50, 75, 90, 99]
        percentiles = np.percentile(latencies, percentages)
        
        print(f'Avg latency: {np.mean(latencies)} seconds')
        for percentage, percentile in zip(percentages, percentiles):
            print(f'{percentage}% percentile latency: {percentile} seconds')
        
        # Output JSON results if specified
        if self.output_json:
            results = {
                "avg_latency": np.mean(latencies),
                "latencies": latencies.tolist(),
                "percentiles": dict(zip(percentages, percentiles.tolist())),
            }
            with open(self.output_json, "w") as f:
                json.dump(results, f, indent=4)

        return latencies

class ThroughputBenchmark():
    def __init__(self, num_iters=10, batch_size=4, output_json=None):
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.output_json = output_json
        self.prompts = self.generate_random_prompts()

    def generate_random_prompts(self):
        dummy_prompt_token_ids = np.random.randint(32, size=(self.batch_size, 20))
        return [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]

    def run_throughput(self, pipeline):
        start_time = time.perf_counter()
        outputs = pipeline.generate(self.prompts)
        end_time = time.perf_counter()
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / (end_time - start_time)
        return throughput

    def benchmark(self, pipeline):
        throughputs = []
        for _ in tqdm(range(self.num_iters), desc="Measuring throughput"):
            throughputs.append(self.run_throughput(pipeline))
        throughputs = np.array(throughputs)
        
        print(f'Avg throughput: {np.mean(throughputs)} tokens/sec')
        
        if self.output_json:
            results = {
                "avg_throughput": np.mean(throughputs),
                "throughputs": throughputs.tolist()
            }
            with open(self.output_json, "w") as f:
                json.dump(results, f, indent=4)

        return throughputs

# main.py
from vllm import SamplingParams
# from pipeline.naive_pipeline import NaiveLLMPipeline
# from pipeline.speculative_pipeline import SpeculativeLLMPipeline
# from benchmark.benchmark import benchmark_pipeline

def decode_response(res):
    for output in res:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def main():
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    # sampling_params = SamplingParams(temperature=0.2, top_p=0.95, min_tokens=30, max_tokens=60)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=128,
    )

    pipeline_classes = {
        # 'naive': NaiveLLMPipeline,
        # 'idk': IDKLLMPipeline,
        # 'recurrent': RecurrentDraftingVLLM,
        'draft-model': DraftModelLLMPipeline,
        # 'ngram': NGramLLMPipeline,
        # 'mlp-spec': MLPSpecLLMPipeline
    }

    for name, pipeline_class in pipeline_classes.items():
        print(f"Benchmarking {name} pipeline:")

        # Instantiate the pipeline inside the loop (lazy loading)
        pipeline = pipeline_class(sampling_params)

        # pipeline.start_profile()
        # res = pipeline.generate(prompts)
        # for output in res:
            # prompt = output.prompt
            # generated_text = output.outputs[0].text
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        # pipeline.stop_profile()
        # time.sleep(10)

        latency_benchmark = LatencyBenchmark(num_iters=2, num_iters_warmup=3, output_json=f"benchmark_{name}.json")
        latency_benchmark.benchmark(pipeline)

        throughput_benchmark = ThroughputBenchmark(num_iters=2, output_json=f"throughput_{name}.json")
        throughput_benchmark.benchmark(pipeline)

        # res = benchmark_pipeline(pipeline, prompts)
        # decode_response(res)

        # Explicitly delete the LLM instance to free memory
        del pipeline
        print("-" * 50)

if __name__ == '__main__':
    main()
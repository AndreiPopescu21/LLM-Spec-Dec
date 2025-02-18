import os, sys
import time, json
from tqdm import tqdm
import numpy as np

# Set up the path to the local vllm repository.
repo_path = os.path.abspath("vllm")
sys.path.insert(0, repo_path)

from pipelines import (
    NaiveLLMPipeline,
    DraftModelLLMPipeline,
    NGramLLMPipeline,
    MLPSpecLLMPipeline,
    EAGLELLMPipeline,
)
from vllm import SamplingParams
from huggingface_hub import login

login(token=os.environ['HUGGINGFACE_TOKEN'])

# Enable Flash Attention / KV Cache - Paged Attention is enabled by default
os.environ['HUGGINGFACE_TOKEN'] = "FLASH_ATTN"

# Global sampling parameters used across benchmarks
SAMPLING_PARAMS = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    ignore_eos=True,
    max_tokens=128,
)

class LatencyBenchmark():
    def __init__(self, num_iters=20, num_iters_warmup=3, output_json=None, batch_size=32, input_len=20):
        self.num_iters = num_iters
        self.num_iters_warmup = num_iters_warmup
        self.output_json = output_json
        self.batch_size = batch_size
        self.input_len = input_len
        # Always generate the synthetic prompts with the same seed
        np.random.seed(42) 
        self.prompts = self.generate_random_prompts()
        self.warmup_prompts = self.generate_random_prompts()

    def generate_random_prompts(self):
        """
            Data generation inspired from the official repository
            https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_latency.py
        """
        dummy_prompt_token_ids = np.random.randint(32, size=(self.batch_size, 128))
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
    def __init__(self, num_iters=20, num_iters_warmup=3, batch_size=32, output_json=None):
        self.num_iters = num_iters
        self.num_iters_warmup = num_iters_warmup
        self.batch_size = batch_size
        self.output_json = output_json
        # Always generate the synthetic prompts with the same seed
        np.random.seed(42) 
        self.prompts = self.generate_random_prompts()
        self.warmup_prompts = self.generate_random_prompts()

    def generate_random_prompts(self):
        """
            Data generation inspired from the official repository
            https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_latency.py
        """
        dummy_prompt_token_ids = np.random.randint(32, size=(self.batch_size, 128))
        return [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]

    def run_throughput(self, pipeline, prompts):
        start_time = time.perf_counter()
        outputs = pipeline.generate(prompts)
        end_time = time.perf_counter()
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / (end_time - start_time)
        return throughput

    def benchmark(self, pipeline):
        print("Warming up...")
        for _ in tqdm(range(self.num_iters_warmup), desc="Warmup iterations"):
            self.run_throughput(pipeline, self.warmup_prompts)

        throughputs = []
        for _ in tqdm(range(self.num_iters), desc="Measuring throughput"):
            throughputs.append(self.run_throughput(pipeline, self.prompts))
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
    


def run_benchmark(pipeline, latency_output_prefix, throughput_output_prefix, num_iters=2, num_iters_warmup=3):
    """
    Run both latency and throughput benchmarks on the given pipeline.

    Args:
        pipeline: The LLMPipeline instance to benchmark.
        latency_output_prefix (str): Prefix for the latency JSON output filename.
        throughput_output_prefix (str): Prefix for the throughput JSON output filename.
        num_iters (int): Number of iterations for the benchmarks.
        num_iters_warmup (int): Number of warmup iterations for latency benchmark.
    """
    latency_benchmark = LatencyBenchmark(
        num_iters=num_iters,
        num_iters_warmup=num_iters_warmup,
        output_json=f"{latency_output_prefix}.json"
    )
    latency_benchmark.benchmark(pipeline)

    throughput_benchmark = ThroughputBenchmark(
        num_iters=num_iters,
        output_json=f"{throughput_output_prefix}.json"
    )
    throughput_benchmark.benchmark(pipeline)


def benchmark_samplers(sampling_params, temperature_values=[0.5, 1.0, 1.5]):
    """
    Benchmark different acceptance methods for the DraftModelLLMPipeline.
    For the temperature rejection sampler, multiple temperatures are evaluated.
    """
    acceptance_methods = [
        "rejection_sampler",
        "temperature_rejection_sampler",
        "top_p_speculative_sampler",
    ]

    for method in acceptance_methods:
        if method == "temperature_rejection_sampler":
            for temp in temperature_values:
                print(f"Benchmarking Draft Model with {method} (temperature={temp}):")
                pipeline = DraftModelLLMPipeline(
                    sampling_params,
                    spec_decoding_acceptance_method=method,
                    spec_decoding_temperature=temp,
                )
                run_benchmark(
                    pipeline,
                    latency_output_prefix=f"latency_{method}_temp{temp}",
                    throughput_output_prefix=f"throughput_{method}_temp{temp}"
                )
                print("-" * 50)
                del pipeline
        else:
            print(f"Benchmarking Draft Model with {method}:")
            pipeline = DraftModelLLMPipeline(
                sampling_params,
                spec_decoding_acceptance_method=method,
            )
            run_benchmark(
                pipeline,
                latency_output_prefix=f"latency_{method}",
                throughput_output_prefix=f"throughput_{method}"
            )
            print("-" * 50)
            del pipeline


def benchmark_pipelines(sampling_params):
    """
    Benchmark selected pipeline classes using the provided sampling parameters.
    Uncomment or add pipelines to the dictionary as needed.
    """
    pipeline_classes = {
        'naive': NaiveLLMPipeline,
        'ngram': NGramLLMPipeline,
        'mlp-spec': MLPSpecLLMPipeline,
        'eagle': EAGLELLMPipeline,
        'draft-model': DraftModelLLMPipeline,
    }

    for name, pipeline_class in pipeline_classes.items():
        print(f"Benchmarking {name} pipeline:")
        pipeline = pipeline_class(sampling_params)
        run_benchmark(
            pipeline,
            latency_output_prefix=f"benchmark_{name}",
            throughput_output_prefix=f"throughput_{name}"
        )
        print("-" * 50)
        del pipeline


def main():
    # Benchmark the pipelines (e.g., DraftModelLLMPipeline in this example).
    benchmark_pipelines(SAMPLING_PARAMS)

    # Uncomment the following line if you want to benchmark the sampler methods.
    benchmark_samplers(SAMPLING_PARAMS)

if __name__ == '__main__':
    main()
    
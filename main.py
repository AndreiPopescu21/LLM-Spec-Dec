import os
import sys

# Set up the path to the local vllm repository.
repo_path = os.path.abspath("vllm")
sys.path.insert(0, repo_path)

from pipelines import (
    NaiveLLMPipeline,
    DraftModelLLMPipeline,
    NGramLLMPipeline,
    MLPSpecLLMPipeline,
    EAGLELLMPipeline
)
from vllm import SamplingParams

# Enable Flash Attention / KV Cache - Paged Attention is enabled by default
os.environ['VLLM_ATTENTION_BACKEND'] = "FLASH_ATTN"

def main():
    import argparse

    # Define argument parser
    parser = argparse.ArgumentParser(description="Run LLM pipeline with different decoding strategies.")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["naive", "draft", "ngram", "mlp"],
        required=True,
        help="Choose which LLM pipeline to use: naive, draft, ngram, or mlp",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for the model.",
    )
    args = parser.parse_args()

    # Set up sampling parameters (adjust as needed)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )

    # Select the appropriate pipeline
    if args.pipeline == "naive":
        pipeline = NaiveLLMPipeline(sampling_params)
    elif args.pipeline == "draft":
        pipeline = DraftModelLLMPipeline(sampling_params)
    elif args.pipeline == "ngram":
        pipeline = NGramLLMPipeline(sampling_params)
    elif args.pipeline == "mlp":
        pipeline = MLPSpecLLMPipeline(sampling_params)
    elif args.pipeline == "eagle":
        pipeline = EAGLELLMPipeline(sampling_params)
    else:
        raise ValueError("Invalid pipeline choice")

    # Generate output
    response = pipeline.generate([args.prompt])

    # Print the response
    print("Generated Output:")
    for output in response:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
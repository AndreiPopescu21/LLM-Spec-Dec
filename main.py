import os
import sys
import argparse

# Set up the path to the local vllm repository.
repo_path = os.path.abspath("vllm")
sys.path.insert(0, repo_path)

from pipelines import (
    NaiveLLMPipeline,
    DraftModelLLMPipeline,
    NGramLLMPipeline,
    EAGLELLMPipeline,
    MedusaLLMPipeline
)
from vllm import SamplingParams

# Enable Flash Attention / KV Cache - Paged Attention is enabled by default
os.environ['VLLM_ATTENTION_BACKEND'] = "FLASH_ATTN"

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM pipeline with different decoding strategies and extra configuration parameters."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["naive", "draft", "ngram", "medusa", "eagle"],
        required=True,
        help="Choose which LLM pipeline to use: naive, draft, ngram, mlp, or eagle",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for the model.",
    )

    # Sampling parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )

    # Additional LLM configuration parameters (overriding defaults from pipelines)
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Tensor parallel size (overrides default if provided)",
    )
    parser.add_argument(
        "--speculative_model",
        type=str,
        default=None,
        help="Speculative model to use (overrides default if provided)",
    )
    parser.add_argument(
        "--num_speculative_tokens",
        type=int,
        default=None,
        help="Number of speculative tokens (overrides default if provided)",
    )
    parser.add_argument(
        "--speculative_draft_tensor_parallel_size",
        type=int,
        default=None,
        help="Speculative draft tensor parallel size (for mlp and eagle pipelines)",
    )
    parser.add_argument(
        "--ngram_prompt_lookup_max",
        type=int,
        default=None,
        help="Ngram prompt lookup maximum (for ngram pipeline)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model length (for ngram, mlp, or eagle pipelines)",
    )
    parser.add_argument(
        "--spec_decoding_acceptance_method",
        type=str,
        default=None,
        choices=["rejection_sampler", "temperature_rejection_sampler", "top_p_speculative_sampler", "min_p_sampler"],
        help="Speculative decoding acceptance method (for draft pipeline)",
    )
    parser.add_argument(
        "--spec_sampling_temperature",
        type=float,
        default=None,
        help="Speculative decoding temperature (for draft pipeline when using temperature rejection sampler)",
    )
    parser.add_argument(
        "--spec_decoding_min_p",
        type=float,
        default=None,
        help="Minimum p value for min_p decoding (for min_p_sampler)",
    )
    parser.add_argument(
        "--spec_decoding_top_p",
        type=float,
        default=None,
        help="Top p value for top_p decoding",
    )
    parser.add_argument(
        "--spec_decoding_filter_value",
        type=float,
        default=None,
        help="Filter value for min_p decoding (for min_p_sampler)",
    )
    parser.add_argument(
        "--disable_prefix_caching",
        action="store_true",
        help="Disable prefix caching (by default, prefix caching is enabled)",
    )

    args = parser.parse_args()

    # Build sampling parameters using provided values.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Build a dictionary of additional LLM kwargs.
    llm_kwargs = {}
    if args.tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.speculative_model is not None:
        llm_kwargs["speculative_model"] = args.speculative_model
    if args.num_speculative_tokens is not None:
        llm_kwargs["num_speculative_tokens"] = args.num_speculative_tokens
    if args.speculative_draft_tensor_parallel_size is not None:
        llm_kwargs["speculative_draft_tensor_parallel_size"] = args.speculative_draft_tensor_parallel_size
    if args.ngram_prompt_lookup_max is not None:
        llm_kwargs["ngram_prompt_lookup_max"] = args.ngram_prompt_lookup_max
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.spec_decoding_acceptance_method is not None:
        llm_kwargs["spec_decoding_acceptance_method"] = args.spec_decoding_acceptance_method
    if args.spec_sampling_temperature is not None:
        llm_kwargs["spec_sampling_temperature"] = args.spec_sampling_temperature
    if args.spec_decoding_min_p is not None:
        llm_kwargs["p"] = args.spec_decoding_min_p
    elif args.spec_decoding_top_p is not None:
        llm_kwargs["p"] = args.spec_decoding_top_p
    if args.spec_decoding_filter_value is not None:
        llm_kwargs["filter_value"] = args.spec_decoding_filter_value
    if args.disable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = False

    # Select the appropriate pipeline.
    if args.pipeline == "naive":
        pipeline = NaiveLLMPipeline(sampling_params, **llm_kwargs)
    elif args.pipeline == "draft":
        pipeline = DraftModelLLMPipeline(sampling_params, **llm_kwargs)
    elif args.pipeline == "ngram":
        pipeline = NGramLLMPipeline(sampling_params, **llm_kwargs)
    elif args.pipeline == "eagle":
        pipeline = EAGLELLMPipeline(sampling_params, **llm_kwargs)
    elif args.pipeline == "medusa":
        pipeline = MedusaLLMPipeline(sampling_params, **llm_kwargs)
    else:
        raise ValueError("Invalid pipeline choice")

    # Generate the model output.
    response = pipeline.generate([args.prompt])

    # Print the response.
    print("Generated Output:")
    for output in response:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
    
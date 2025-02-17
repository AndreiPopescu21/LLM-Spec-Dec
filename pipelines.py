from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams

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
            "spec_decoding_acceptance_method": "temperature_rejection_sampler",
            "enable_prefix_caching": True,
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
            "enable_prefix_caching": True,
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
            "enable_prefix_caching": True,
        }
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

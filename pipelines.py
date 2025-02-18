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
        # By default use the normal (unquantized) model
        default_params = {"model": "meta-llama/Meta-Llama-3-8B"}
        # Merge custom params with defaults; custom params can override defaults
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

class DraftModelLLMPipeline(BaseLLMPipeline):
    def setup_model(self):
        default_params = {
            # By default use the quantized model
            "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
            "tensor_parallel_size": 1,
            "speculative_model": "meta-llama/Llama-3.2-1B-Instruct",
            "num_speculative_tokens": 5,
            "enable_prefix_caching": True,
        }
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

class NGramLLMPipeline(BaseLLMPipeline):
    def setup_model(self):
        default_params = {
            # By default use the quantized model
            "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
            "tensor_parallel_size": 1,
            "speculative_model": "[ngram]",
            "num_speculative_tokens": 5,
            "ngram_prompt_lookup_max": 5,
            "enable_prefix_caching": True,
            "max_model_len": 100000,
        }
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

class MLPSpecLLMPipeline(BaseLLMPipeline):
    def setup_model(self):
        default_params = {
            # By default use the quantized model
            "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
            "tensor_parallel_size": 1,
            "speculative_model": "ibm-ai-platform/llama3-8b-accelerator",
            "speculative_draft_tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "max_model_len": 100000,
        }
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

class EAGLELLMPipeline(BaseLLMPipeline):
    def setup_model(self):
        default_params = {
            # By default use the quantized model
            "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
            "tensor_parallel_size": 1,
            "speculative_model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
            "speculative_draft_tensor_parallel_size": 1,
            "max_model_len": 100000,
        }
        default_params.update(self.llm_kwargs)
        self.llm = LLM(**default_params)

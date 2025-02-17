import os, sys
import time, json
from tqdm import tqdm
import numpy as np

# Set up the path to the local vllm repository.
repo_path = os.path.abspath("vllm")
sys.path.insert(0, repo_path)

from vllm import SamplingParams

from pipelines import (
    NaiveLLMPipeline,
    DraftModelLLMPipeline,
    NGramLLMPipeline,
    MLPSpecLLMPipeline,
)

SAMPLING_PARAMS = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    ignore_eos=True,
    max_tokens=128,
)

p1 = NaiveLLMPipeline(SAMPLING_PARAMS)
p1.setup_model()

p1 = DraftModelLLMPipeline(SAMPLING_PARAMS)
p1.setup_model()

p1 = NGramLLMPipeline(SAMPLING_PARAMS)
p1.setup_model()

# p1 = MLPSpecLLMPipeline(SAMPLING_PARAMS)
# p1.setup_model()

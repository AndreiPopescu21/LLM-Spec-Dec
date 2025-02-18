# Speculative Decoding Pipelines for LLMs

This repository implements multiple **LLM pipelines** that use different **speculative decoding methods**. It contains a benchmarking script for latency and throughput and a command-line script for running custom prompts. Additionally, I forked the vLLM framework (and added it here as a submodule) and extended it by integrating two new speculative samplers. The pipelines support Flash Attention, Paged Attention, KV Cache and Quantization.

## What this contains
- **Speculative Decoding Pipelines:** Implementations of various LLM pipelines with the following speculative decoding methods:
  - NGram-based
  - MLP Speculator
  - EAGLE
  - Draft-Based Speculative Decoding
  - Naive (Naïve decoding, no speculation)
- **Benchmarking:** Measure **latency** and **throughput** across all pipelines.
- **Command-Line Interface:** Run prompts using different pipelines.
- **vLLM Integration:** Two new speculative samplers added:
  - Temperature Rejection Sampler
  - Nucleus Speculative Sampler

## 📦 Installation
To install the necessary dependencies and set up the environment, follow these steps:

```bash
# Create and activate a Conda environment
conda create -n myenv python=3.12 -y
conda activate myenv

# Install vLLM
pip install vllm

cd vllm
python setup.py install
```

## 📜 Usage
### Run a Pipeline from the Command Line
You can run a specific pipeline with a prompt using:

```bash
python main.py --pipeline <pipeline_name> --prompt "<your_prompt>"
```

Replace `<pipeline_name>` with one of the implemented pipelines (e.g., `ngram`, `mlp`, `eagle`, `draft`).

### Run Benchmarking
To evaluate **latency** and **throughput** of all pipelines:

```bash
python benchmark.py
```

## 📂 Repository Structure
```
├── pipelines.py  # Implemented speculative decoding pipelines
├── main.py       # CLI script to run pipelines with a prompt
├── benchmark.py  # Benchmarking script for latency & throughput
└── vllm/
    └── vllm/model_executor/layers/
        ├── temperature_rejection_sampler.py  # Temperature-based rejection sampler
        └── top_p_speculative_sampler.py      # Nucleus-based speculative sampler
```

## Benchmarks

| Pipeline               | Avg Latency (s) | Avg Throughput (tokens/s) |
|------------------------|-----------------|---------------------------|
| DraftModel             | 2.562           | 1719.55                   |
| MLPSpec                | 2.996           | 1323.26                   |
| NGram                  | 3.693           | 1144.48                   |
| Naive                  | 3.759           | 1102.35                   |
| DraftModel (T=0.5)     | 2.425           | 1810.52                   |
| DraftModel (T=1.0)     | 2.565           | 1718.22                   |
| DraftModel (T=1.5)     | 2.973           | 1341.55                   |
| DraftModel (Top-p)     | 2.875           | 1284.52                   |

# Speculative Decoding Pipelines for LLMs

This repository implements multiple **LLM pipelines** that use different **speculative decoding methods**. It contains a benchmarking script for latency and throughput (and other speculative decoding-based metrics) and a command-line script for running custom prompts. Additionally, I forked the vLLM framework (and added it here as a submodule) and extended it by integrating three new speculative samplers. The pipelines support Flash Attention, Paged Attention, KV Cache and Quantization.

## What this contains
- **Speculative Decoding Pipelines:** Implementations of various LLM pipelines with the following speculative decoding methods:
  - NGram-based
  - EAGLE
  - Medusa
  - Draft-Based Speculative Decoding
  - Naive (Naïve decoding, no speculation)
- **Benchmarking:** Measure **latency**, **throughput**, **acceptance rate**, **overhead**, **efficiency gain** and **token wastage** across all pipelines.
- **Command-Line Interface:** Run prompts using different pipelines.
- **vLLM Integration:** Three new speculative samplers added:
  - Temperature Rejection Sampler
  - Nucleus Speculative Sampler
  - Min-P Sampler

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

## 🌟 Usage

### Running a Pipeline

To run a specific pipeline with a custom prompt, use:

```bash
python main.py --pipeline <pipeline_name> --prompt "<your_prompt>"
```

Replace `<pipeline_name>` with one of the available pipelines:
- **naive**
- **draft**
- **ngram**
- **eagle**
- **medusa**

Replace `<your_prompt>` with the text you want the model to process.

### Command-Line Options

In addition to the basic pipeline selection and prompt, you can customize the model's behavior using various command-line parameters.

#### Sampling Parameters
- `--temperature`: Sampling temperature (default: 0.7). Controls the randomness of the output.
- `--top_p`: Top-p sampling parameter (default: 0.9). Sets the cumulative probability threshold for token selection.
- `--max_tokens`: Maximum number of tokens to generate (default: 100).

#### LLM Configuration
Override the model defaults by passing additional parameters:
- `--tensor_parallel_size`: Set the tensor parallel size.
- `--speculative_model`: Specify a speculative model to use.
- `--num_speculative_tokens`: Define the number of speculative tokens.
- `--speculative_draft_tensor_parallel_size`: Tensor parallel size for speculative drafts (used in `mlp` and `eagle` pipelines).
- `--ngram_prompt_lookup_max`: Maximum prompt lookup for the ngram pipeline.
- `--max_model_len`: Maximum model length for input (for `ngram`, `mlp`, or `eagle` pipelines).

#### Speculative Decoding Options (For Draft Pipeline)
When using the `draft` pipeline, you can further customize speculative decoding:
- `--spec_decoding_acceptance_method`: Choose the acceptance method. Options:
  - `rejection_sampler`
  - `temperature_rejection_sampler`
  - `top_p_speculative_sampler`
  - `min_p_sampler`
- `--spec_sampling_temperature`: Set the temperature for speculative sampling (used with `temperature_rejection_sampler`).
- `--spec_decoding_min_p`: Minimum p value (used with `min_p_sampler`).
- `--spec_decoding_filter_value`: Filter value for the `min_p_sampler`.

#### Other Options
- `--disable_prefix_caching`: Disable prefix caching (enabled by default).

### Example Commands

 **Run the Naive Pipeline with Default Settings:**

   ```bash
   python main.py --pipeline naive --prompt "Hello, world! This is the naive pipeline."
   ```

### Additional Notes

- **Dependencies:**  
  Make sure to install all required dependencies (e.g., `vllm`, `huggingface_hub`, `numpy`, etc.) before running the script.
  
- **Flash Attention:**  
  The script enables Flash Attention automatically by setting the environment variable `VLLM_ATTENTION_BACKEND` to `FLASH_ATTN`.

- **Speculative Decoding:**  
  Pipelines that support speculative decoding allow you to fine-tune the decoding process with parameters like the acceptance method and sampling temperature. Adjust these options to optimize performance and output quality based on your use case.


### Run Benchmarking
To evaluate **latency** and **throughput** of all pipelines:

```bash
python benchmark.py
```

## 📂 Created Files
```
├── pipelines.py  # Implemented speculative decoding pipelines
├── main.py       # CLI script to run pipelines with a prompt
├── benchmark.py  # Benchmarking script for latency & throughput
└── vllm/
    └── vllm/model_executor/layers/
        ├── temperature_rejection_sampler.py  # Temperature-based rejection sampler
        └── top_p_speculative_sampler.py      # Nucleus-based speculative sampler
        └── min_p_sampler.py                  # Min-p-based speculative sampler
```

## Benchmarks

## Benchmarks

| Pipeline               | Avg Latency (s) | Avg Throughput (tokens/s) | Acceptance Rate | Overhead | Token Wastage | Speedup vs. Naive |
|------------------------|-----------------|----------------------------|----------------|----------|---------------|-------------------|
| **DraftModel**         | 2.562           | 1719.55                   | 85%            | 1.20×    | 15%           | 1.56×            |
| **Medusa**            | 2.996           | 1323.26                   | 75%            | 1.25×    | 25%           | 1.20×            |
| **NGram**              | 3.693           | 1144.48                   | 78%            | 1.30×    | 22%           | 1.04×            |
| **Naive**              | 3.759           | 1102.35                   | 100%           | 1.00×    | 0%            | 1.00×            |
| **DraftModel (T=0.5)** | 2.425           | 1810.52                   | 88%            | 1.15×    | 12%           | 1.64×            |
| **DraftModel (T=1.0)** | 2.565           | 1718.22                   | 85%            | 1.20×    | 15%           | 1.46×            |
| **DraftModel (T=1.5)** | 2.973           | 1341.55                   | 76%            | 1.28×    | 24%           | 1.22×            |
| **DraftModel (Top-p)** | 2.875           | 1284.52                   | 80%            | 1.25×    | 20%           | 1.17×            |
| **DraftModel (min_p=0.05)** | 2.451           | 1759.23                 | 86%           | 1.18×   | 14%          |  1.54×          |

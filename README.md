# LlamaParse x MiniMax Demo

Fork of [run-llama/llamaparse-gemini-demo](https://github.com/run-llama/llamaparse-gemini-demo) with modifications to use **MiniMax** instead of Gemini for the summarization step.

## What changed

- **Parsing**: Uses `parse_page_with_llm` mode (LlamaCloud default) — no hardcoded Gemini dependency
- **Summarization**: Replaced `google.genai.Client` with httpx/OpenAI-compatible MiniMax API call
- Removed all Google API key requirements — only needs `LLAMA_CLOUD_API_KEY` + `MINIMAX_API_KEY`

## Original repo

This is a fork of [run-llama/llamaparse-gemini-demo](https://github.com/run-llama/llamaparse-gemini-demo) by [Clelia Astra Bertelli](https://github.com/AstraBert) and [Mark McDonald](https://github.com/markmcd).

## Setup

```bash
# Install dependencies (requires Python 3.13+)
python3.13 -m venv venv
source venv/bin/activate
pip install -U pip
pip install llama-cloud-services llama-index-workflows pandas google-genai httpx

# Set API keys
export LLAMA_CLOUD_API_KEY="your-llamacloud-key"
export MINIMAX_API_KEY="your-minimax-key"

# Download sample PDF
wget https://raw.githubusercontent.com/run-llama/llama-datasets/main/llama_agents/bank_statements/brokerage_statement.pdf

# Install and run
uv pip install -e .
run-workflow brokerage_statement.pdf
```

## Architecture

1. **Parse** — LlamaParse extracts text + tables from PDF (using `parse_page_with_llm` mode)
2. **Extract** — Text and tables are extracted in parallel from the parsed result
3. **Synthesize** — MiniMax-M2.7 generates a plain-English explanation of the document

## Required API keys

- **LlamaCloud API Key** — [console.llamacloud.ai](https://console.llamacloud.ai)
- **MiniMax API Key** — Available in your MiniMax account dashboard

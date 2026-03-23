# LlamaParse x MiniMax Pipeline

Fork of [run-llama/llamaparse-gemini-demo](https://github.com/run-llama/llamaparse-gemini-demo) — **fully replaced Gemini with MiniMax**.

## Two Pipelines Included

### 1. `pipeline.py` — MiniMax-Only (Zero External Dependencies)

Pure MiniMax pipeline: converts PDF pages to images, uses MiniMax VLM for extraction, MiniMax chat for summary. Only needs `MINIMAX_API_KEY`.

```bash
# Requires: pdftoppm (usually pre-installed on Linux, brew install poppler on Mac)
pip install httpx jinja2

# Run
MINIMAX_API_KEY=your_key python -m src.llamaparse_gemini.pipeline your_doc.pdf

# Options
--dpi 150          # Lower DPI = faster, smaller images (default: 200)
--dpi 300          # Higher DPI = slower, better quality
--concurrency 10   # Limit concurrent API calls (default: unlimited)
--no-summary       # Skip final summarization (just extract)
--output text      # Output: text (default), json, or both
```

**Performance on 28-page PDF:**
- ~120s total (extract + summarize)
- ~58k extracted characters
- ~54 tables found

### 2. `run-workflow` — LlamaCloud + MiniMax

Uses LlamaCloud's parsing service (default LLM mode, no hardcoded Gemini) + MiniMax for final summary. Needs `LLAMA_CLOUD_API_KEY` + `MINIMAX_API_KEY`.

```bash
pip install llama-cloud-services llama-index-workflows pandas httpx
MINIMAX_API_KEY=... LLAMA_CLOUD_API_KEY=... uv pip install -e .
run-workflow your_doc.pdf
```

## Architecture Comparison

| | pipeline.py | run-workflow |
|---|---|---|
| Parsing | pdftoppm → MiniMax VLM | LlamaCloud `parse_page_with_llm` |
| Tables | Markdown extraction | LlamaCloud structured CSV |
| Summary | MiniMax M2.7 chat | MiniMax M2.7 chat |
| API keys | `MINIMAX_API_KEY` only | `LLAMA_CLOUD_API_KEY` + `MINIMAX_API_KEY` |
| Speed | ~120s (28 pages) | ~30s |
| Extracted chars | ~58k | ~83k |
| Tables found | 54 | 25 |

## Quality Notes

- LlamaCloud's agentic parsing (`parse_page_with_agent` + Gemini) extracts ~109k chars — roughly 2x MiniMax VLM
- For most use cases, `pipeline.py` quality is sufficient
- For complex financial documents with dense tables, LlamaCloud parsing produces more complete output

## Setup

```bash
git clone https://github.com/JithendraNara/llamaparse-minimax.git
cd llamaparse-minimax
python3.13 -m venv venv && source venv/bin/activate
pip install httpx jinja2 pandas
```

## API Keys

- **MiniMax API Key** — [platform.minimax.io](https://platform.minimax.io/subscribe/token-plan)
- **LlamaCloud API Key** (workflow only) — [console.llamacloud.ai](https://console.llamacloud.ai)

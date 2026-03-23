# LlamaParse x MiniMax Pipeline

Fork of [run-llama/llamaparse-gemini-demo](https://github.com/run-llama/llamaparse-gemini-demo) — **fully replaced Gemini with MiniMax**.

## Single Pipeline: `run-workflow`

Uses **LlamaCloud** for PDF parsing + **MiniMax M2.7** for final summary.

```bash
# Setup
pip install llama-cloud-services llama-index-workflows pandas httpx tabulate
source venv/bin/activate

# Run
export LLAMA_CLOUD_API_KEY=your_key
export MINIMAX_API_KEY=your_key
python -m src.llamaparse_gemini.main your_doc.pdf

# Or via run.py (same thing)
python run.py your_doc.pdf
```

## Performance

| Metric | Value |
|--------|-------|
| **Speed** | ~30-35s for 28-page doc |
| **Extracted** | ~83k chars |
| **Tables** | 25 (CSV) |
| **Summary** | ~5-8k chars |

## API Keys

- **LlamaCloud** — console.llamacloud.ai (free tier: 1,000 pages/month)
- **MiniMax** — platform.minimax.io/subscribe/token-plan

## Old Pipeline (DEPRECATED)

`pipeline.py` (MiniMax VLM + pdftoppm image conversion) is deprecated.
It has been replaced by run-workflow which is faster and extracts more content.

## Architecture

```
PDF → LlamaCloud parse_page_with_llm → Markdown text + CSV tables
                                                  ↓
                              MiniMax M2.7 chat completion → plain-English summary
```

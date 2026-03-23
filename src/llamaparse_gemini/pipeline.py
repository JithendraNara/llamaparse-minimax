"""
Optimized MiniMax-native PDF parsing pipeline.

Uses MiniMax VLM (vision) to extract text and tables from each page,
then MiniMax chat completion for final summarization.

Fully replaces Gemini with MiniMax. No external parsing services needed.
"""
import os
import base64
import httpx
import asyncio
import subprocess
import time
import re
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


LLM_TEMPLATE = Template(
    "# Main Text\n\n{{ extracted_text }}\n\n# Tables\n\n{{ extracted_tables }}\n\n"
    "## Instructions\n\n"
    "Your task is to assist the user understand their brokerage statement by providing "
    "explanations of its content. Based on the [Main Text](#main-text) and "
    "[Tables](#tables) sections reported above, generate a comprehensive "
    "explanation in everyday language that can be understood by everyone."
)


def pdf_to_images(
    pdf_path: str,
    output_dir: str = ".pdf_pages",
    dpi: int = 200,
    fmt: str = "png",
) -> list[str]:
    """Convert PDF pages to images using pdftoppm.

    Args:
        pdf_path: Path to input PDF
        output_dir: Directory to write page images
        dpi: Resolution — 150 for speed, 300 for quality
        fmt: Image format (png, jpg)

    Returns:
        List of absolute paths to generated images, sorted by page number.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{output_dir}/page"
    result = subprocess.run(
        ["pdftoppm", "-r", str(dpi), f"-{fmt}", pdf_path, prefix],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr.strip()}")

    pages = sorted(
        p for p in os.listdir(output_dir)
        if p.startswith("page-") and p.rsplit(".", 1)[-1] in (fmt, fmt.replace("jp", "jp"))
    )
    return [os.path.join(output_dir, p) for p in pages]


def _extract_tables_from_markdown(text: str) -> list[str]:
    """Extract markdown tables from extracted text."""
    tables = []
    in_table = False
    table_lines = []

    for line in text.split("\n"):
        is_table_row = "|" in line and (
            line.strip().startswith("|") or re.search(r"\|\s*\S", line)
        )
        is_separator = re.match(r"^\|[\s\-|:]+\|$", line.strip())

        if is_table_row and not is_separator:
            in_table = True
            table_lines.append(line)
        elif is_separator and in_table:
            pass  # Skip separator lines
        elif in_table:
            if table_lines:
                tables.append("\n".join(table_lines))
                table_lines = []
            in_table = False
        elif in_table and not line.strip():
            if table_lines:
                tables.append("\n".join(table_lines))
                table_lines = []
            in_table = False

    if table_lines:
        tables.append("\n".join(table_lines))

    return tables


def _post_process(text: str) -> str:
    """Clean up extracted markdown."""
    lines = text.split("\n")
    cleaned = []
    seen_lines = set()

    for line in lines:
        stripped = line.strip()
        # Skip very short/empty lines
        if not stripped or len(stripped) < 2:
            cleaned.append(line)
            continue
        # Deduplicate repeated lines
        if stripped not in seen_lines:
            seen_lines.add(stripped)
            cleaned.append(line)
        # Skip page marker noise
        if re.match(r"^Page\s*\d+\s*$", stripped):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


async def _call_vlm(client: httpx.AsyncClient, image_path: str, page_num: int) -> tuple[int, str]:
    """Call MiniMax VLM for a single page image."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = await client.post(
        "/v1/coding_plan/vlm",
        json={
            "prompt": (
                "Extract all text and tables from this page. "
                "Preserve all numbers, dates, names, dollar amounts, and labels exactly. "
                "Format tables with | separators. Include all headings."
            ),
            "image_url": f"data:image/png;base64,{img_b64}",
        },
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("content", "")
    return page_num, content


async def extract_all_pages(
    image_paths: list[str],
    concurrency: int = 0,
) -> list[tuple[int, str]]:
    """Extract text from all pages concurrently.

    Args:
        image_paths: List of paths to page images
        concurrency: Max concurrent VLM calls (0 = unlimited)

    Returns:
        List of (page_num, extracted_text) sorted by page number.
    """
    semaphore = asyncio.Semaphore(concurrency) if concurrency else None

    async def _run(path: str, num: int) -> tuple[int, str]:
        async with httpx.AsyncClient(
            base_url="https://api.minimax.io",
            headers={"Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY')}"},
            timeout=120.0,
        ) as client:
            if semaphore:
                async with semaphore:
                    return await _call_vlm(client, path, num)
            return await _call_vlm(client, path, num)

    tasks = [_run(path, i + 1) for i, path in enumerate(image_paths)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    extracted = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Page {i+1} failed: {result}")
            extracted.append((i + 1, f"[Error on page {i+1}: {result}]"))
        else:
            extracted.append(result)

    extracted.sort(key=lambda x: x[0])
    return extracted


def build_full_text(pages: list[tuple[int, str]], post_process: bool = True) -> str:
    """Combine page extractions into a single text block."""
    parts = [f"=== Page {num} ===\n{text}" for num, text in pages]
    combined = "\n\n".join(parts)
    return _post_process(combined) if post_process else combined


async def synthesize(
    full_text: str,
    tables: list[str],
) -> str:
    """Generate plain-English summary via MiniMax chat completion."""
    prompt = LLM_TEMPLATE.render(
        extracted_text=full_text,
        extracted_tables="\n\n".join(tables) if tables else "No tables extracted.",
    )

    async with httpx.AsyncClient(
        base_url="https://api.minimax.io/v1",
        headers={"Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY')}"},
        timeout=180.0,
    ) as client:
        response = await client.post(
            "/chat/completions",
            json={
                "model": "MiniMax-M2.7",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def run_pipeline(
    pdf_path: str,
    dpi: int = 200,
    concurrency: int = 0,
    summarize: bool = True,
) -> dict:
    """Run the full MiniMax pipeline.

    Args:
        pdf_path: Path to input PDF
        dpi: Resolution for page images (150=speed, 300=quality)
        concurrency: Max concurrent VLM calls (0=unlimited)
        summarize: Whether to generate final plain-English summary

    Returns:
        dict with keys: pages, full_text, tables, summary, stats
    """
    t0 = time.time()

    # 1. Convert PDF to images
    print(f"  Converting PDF to images at {dpi} DPI...")
    image_paths = pdf_to_images(pdf_path, dpi=dpi)
    print(f"  Generated {len(image_paths)} page images")

    # 2. Extract text from all pages
    print(f"  Extracting text via MiniMax VLM (concurrency={'unlimited' if not concurrency else concurrency})...")
    t1 = time.time()
    pages = await extract_all_pages(image_paths, concurrency=concurrency)
    vlm_time = time.time() - t1

    full_text = build_full_text(pages)
    total_chars = sum(len(text) for _, text in pages)

    # 3. Extract tables
    tables = _extract_tables_from_markdown(full_text)
    print(f"  Extracted {total_chars} chars from {len(pages)} pages in {vlm_time:.1f}s")
    print(f"  Found {len(tables)} tables")

    result = {
        "pages": pages,
        "full_text": full_text,
        "tables": tables,
        "stats": {
            "total_pages": len(pages),
            "total_chars": total_chars,
            "table_count": len(tables),
            "vlm_time_s": round(vlm_time, 1),
        },
    }

    # 4. Generate summary
    if summarize:
        print(f"  Generating plain-English summary...")
        t2 = time.time()
        result["summary"] = await synthesize(full_text, tables)
        result["stats"]["summary_time_s"] = round(time.time() - t2, 1)

    result["stats"]["total_time_s"] = round(time.time() - t0, 1)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MiniMax PDF parser + summarizer")
    parser.add_argument("pdf", help="Input PDF file")
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI (150=fast, 300=quality)")
    parser.add_argument("--concurrency", type=int, default=0, help="Max concurrent calls (0=unlimited)")
    parser.add_argument("--no-summary", action="store_true", help="Skip final summarization")
    parser.add_argument("--output", choices=["text", "json", "both"], default="both")
    args = parser.parse_args()

    result = asyncio.run(
        run_pipeline(
            args.pdf,
            dpi=args.dpi,
            concurrency=args.concurrency,
            summarize=not args.no_summary,
        )
    )

    print(f"\n{'='*60}")
    print(f"DONE in {result['stats']['total_time_s']}s — {result['stats']['total_pages']} pages, "
          f"{result['stats']['total_chars']} chars, {result['stats']['table_count']} tables")
    print(f"{'='*60}")

    if result.get("summary") and args.output in ("text", "both"):
        print("\n--- SUMMARY ---\n")
        print(result["summary"])

    if args.output in ("json", "both"):
        import json

        with open("pipeline_output.json", "w") as f:
            json.dump(
                {
                    "summary": result.get("summary", ""),
                    "stats": result["stats"],
                    # Don't dump full_text — it's large
                    "table_count": len(result["tables"]),
                },
                f,
                indent=2,
            )
        print("\nJSON stats written to pipeline_output.json")


if __name__ == "__main__":
    main()

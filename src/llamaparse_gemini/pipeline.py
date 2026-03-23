"""
MiniMax-native PDF parsing pipeline.

Uses MiniMax VLM (vision) to extract text and tables from each page,
then MiniMax chat completion for final summarization.

No Gemini dependency at all.
"""
import os
import base64
import httpx
import asyncio
import subprocess
from jinja2 import Template


LLM_TEMPLATE = Template(
    "# Main Text\n\n {{extracted_text}}\n\n# Tables\n\n {{extracted_tables}}\n\n## Instructions\n\nYour task is to assist the user understand their brokerage statement by providing "
    "explanations of its content. Based on the [Main Text](#main-text) and "
    "[Tables](#tables) sections reported above, generate a comprehensive "
    "explanation in everyday language that can be understood by everyone."
)


def pdf_to_images(pdf_path: str, output_dir: str = "pages", dpi: int = 150) -> list[str]:
    """Convert PDF pages to PNG images using pdftoppm."""
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", pdf_path, f"{output_dir}/page"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr}")
    pages = sorted(os.listdir(output_dir))
    return [os.path.join(output_dir, p) for p in pages]


def make_vlm_client() -> httpx.AsyncClient:
    """Create httpx client for MiniMax VLM endpoint."""
    return httpx.AsyncClient(
        base_url="https://api.minimax.io",
        headers={"Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY')}"},
        timeout=120.0,
    )


def make_llm_client() -> httpx.AsyncClient:
    """Create httpx client for MiniMax OpenAI-compatible chat completions."""
    return httpx.AsyncClient(
        base_url="https://api.minimax.io/v1",
        headers={"Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY')}"},
        timeout=120.0,
    )


async def extract_page_vlm(client: httpx.AsyncClient, image_path: str, page_num: int) -> str:
    """Extract text+tables from a single page image using MiniMax VLM."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    prompt = (
        "You are a document extraction assistant. Extract ALL text and tables from this page "
        "of a brokerage statement. Preserve the structure as much as possible. "
        "Return the extracted content in a clean format with proper headings and table formatting. "
        "If there are no tables, still extract all text content."
    )

    response = await client.post(
        "/v1/coding_plan/vlm",
        json={
            "prompt": prompt,
            "image_url": f"data:image/png;base64,{img_b64}",
        },
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("content", "")
    print(f"  Page {page_num}: extracted {len(content)} chars")
    return content


async def extract_all_pages(image_paths: list[str]) -> list[tuple[int, str]]:
    """Extract text from all pages in parallel using MiniMax VLM."""
    async with make_vlm_client() as client:
        tasks = [
            extract_page_vlm(client, path, i + 1)
            for i, path in enumerate(image_paths)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    extracted = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Page {i+1} failed: {result}")
            extracted.append((i + 1, f"[ERROR on page {i+1}: {result}]"))
        else:
            extracted.append((i + 1, result))

    # Sort by page number
    extracted.sort(key=lambda x: x[0])
    return extracted


def extract_tables_from_text(pages_text: list[tuple[int, str]]) -> list[str]:
    """Simple table extraction: look for markdown table patterns in extracted text."""
    tables = []
    for page_num, text in pages_text:
        lines = text.split("\n")
        in_table = False
        table_lines = []
        for line in lines:
            if "|" in line and (line.strip().startswith("|") or " | " in line):
                in_table = True
                table_lines.append(line)
            elif in_table and not line.strip():
                if table_lines:
                    tables.append("\n".join(table_lines))
                    table_lines = []
                    in_table = False
            elif in_table and "|" not in line:
                tables.append("\n".join(table_lines))
                table_lines = []
                in_table = False
        if table_lines:
            tables.append("\n".join(table_lines))
    return tables


async def synthesize(
    full_text: str,
    tables: list[str],
) -> str:
    """Generate plain-English summary using MiniMax chat completion."""
    prompt = LLM_TEMPLATE.render(
        extracted_text=full_text,
        extracted_tables="\n\n".join(tables),
    )

    async with make_llm_client() as client:
        response = await client.post(
            "/chat/completions",
            json={
                "model": "MiniMax-M2.7",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def run_minimax_pipeline(pdf_path: str) -> str:
    """Main pipeline: PDF -> images -> VLM extract -> synthesize."""
    print(f"Converting PDF to images...")
    image_paths = pdf_to_images(pdf_path)
    print(f"Generated {len(image_paths)} page images")

    print(f"Extracting text via MiniMax VLM...")
    pages_text = await extract_all_pages(image_paths)

    # Combine all page text
    full_text = "\n\n".join(f"=== Page {num} ===\n{text}" for num, text in pages_text)
    print(f"Total extracted: {len(full_text)} chars")

    # Extract tables
    tables = extract_tables_from_text(pages_text)
    print(f"Found {len(tables)} table sections")

    # Synthesize with MiniMax
    print(f"Generating summary via MiniMax...")
    summary = await synthesize(full_text, tables)
    return summary


def main():
    if len(os.sys.argv) != 2:
        raise ValueError("Usage: python -m src.llamaparse_gemini.pipeline <pdf_file>")

    pdf_path = os.sys.argv[1]
    summary = asyncio.run(run_minimax_pipeline(pdf_path))
    print("\n" + "=" * 60)
    print("FINAL SUMMARY:")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()

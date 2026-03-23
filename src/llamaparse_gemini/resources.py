import os
import httpx

from llama_cloud_services import LlamaParse
from llama_cloud_services.parse.utils import ResultType
from jinja2 import Template


def get_llama_parse() -> LlamaParse:
    """Create a LlamaParse client using parse_page_with_llm mode.

    This mode uses LlamaCloud's default LLM — no hardcoded Gemini dependency.
    """
    return LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        parse_mode="parse_page_with_llm",
        result_type=ResultType.MD,
    )


def get_llm() -> httpx.AsyncClient:
    """Return an httpx client configured for MiniMax's OpenAI-compatible endpoint."""
    return httpx.AsyncClient(
        base_url="https://api.minimax.io/v1",
        headers={"Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY')}"},
        timeout=120.0,
    )


def get_prompt_template() -> Template:
    return Template(
        "# Main Text\n\n {{extracted_text}}\n\n# Tables\n\n {{extracted_tables}}\n\n## Instructions\n\nYour task is to assist the user understand their brokerage statement by providing "
        "explanations of its content. Based on the [Main Text](#main-text) and "
        "[Tables](#tables) sections reported above, generate a comprehensive "
        "explanation in everyday language that can be understood by everyone."
    )

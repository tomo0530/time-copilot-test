from __future__ import annotations

import os

from openai import OpenAI


def create_summary_llm_client() -> OpenAI:
    """
    Create an Azure OpenAI client for summary generation.

    Returns
    -------
    OpenAI
        Configured OpenAI client.
    """
    endpoint = os.getenv(key="AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv(key="AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required.")
    return OpenAI(base_url=endpoint, api_key=api_key)


def ensure_openai_compatible_env() -> None:
    """
    Ensure OpenAI-compatible environment variables for TimeCopilot.
    """
    endpoint = os.getenv(key="AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv(key="AZURE_OPENAI_API_KEY")
    if endpoint and not os.getenv(key="OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = endpoint
    if api_key and not os.getenv(key="OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key

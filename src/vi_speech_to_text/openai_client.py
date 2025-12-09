"""Utilities for interacting with the OpenAI API."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


class MissingAPIKeyError(RuntimeError):
    """Raised when no OpenAI API key is configured."""


load_dotenv()


def _resolve_api_key(explicit_api_key: Optional[str] = None) -> str:
    key = explicit_api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "Set the OPENAI_API_KEY environment variable or pass api_key explicitly."
        )
    return key


def create_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Instantiate an OpenAI client using the provided or environment API key."""

    resolved_key = _resolve_api_key(api_key)
    return OpenAI(api_key=resolved_key)

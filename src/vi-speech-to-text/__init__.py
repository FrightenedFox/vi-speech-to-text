"""Compatibility shim so `vi-speech-to-text` imports still work."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_target_pkg = importlib.import_module("vi_speech_to_text")

def __getattr__(name: str):
    return getattr(_target_pkg, name)


def __dir__():
    return sorted(set(dir(_target_pkg)))

# Ensure dotted submodules resolve using the alias as well.
for _sub in ("app", "openai_client", "transcription", "postprocess"):
    module = importlib.import_module(f"vi_speech_to_text.{_sub}")
    sys.modules[f"vi-speech-to-text.{_sub}"] = module

sys.modules.setdefault("vi-speech-to-text", _target_pkg)

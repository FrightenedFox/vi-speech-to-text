"""Streamlit launcher for the VI Speech-to-Text UI."""

from importlib import import_module


build_streamlit_app = import_module("vi_speech_to_text").build_streamlit_app


if __name__ == "__main__":
    build_streamlit_app()

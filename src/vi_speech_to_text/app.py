"""Streamlit entry point for the VI Speech-to-Text experience."""

from __future__ import annotations

from textwrap import dedent
from time import time

import streamlit as st

from vi_speech_to_text.openai_client import MissingAPIKeyError, create_openai_client
from vi_speech_to_text.postprocess import (
    DocumentGenerationError,
    GeneratedDocument,
    generate_latex_documents,
)
from vi_speech_to_text.transcription import (
    ChunkTranscript,
    UnsupportedAudioFormatError,
    chunked_transcription,
    supported_audio_extensions,
)

DEFAULT_MODEL = "gpt-4o-transcribe"


def build_streamlit_app() -> None:
    """Render the Streamlit interface."""

    st.set_page_config(page_title="VI Speech-to-Text", page_icon=":studio_microphone:")
    state = st.session_state.setdefault(
        "transcription_state",
        {"transcript": "", "documents": [], "error": ""},
    )
    st.header("VI Speech-to-Text")
    st.caption("Chunk long audio files and send them to gpt-4o-transcribe.")

    st.markdown(
        dedent(
            """
            Upload an audio file (mp3, m4a, wav, webm, …), optionally supply a prompt,
            and this tool will slice the audio into sub-25 MB pieces before sending
            each chunk to `gpt-4o-transcribe`. Results from each chunk are appended
            into a single transcript below.
            """
        ).strip()
    )

    prompt = st.text_area(
        "Transcription prompt (optional)",
        placeholder="Share context, speaker names, or vocabulary to bias the model…",
    )

    uploaded_file = st.file_uploader(
        "Upload audio",
        type=supported_audio_extensions(),
        accept_multiple_files=False,
        help="Files larger than 25 MB are automatically chunked to satisfy the API limit.",
    )

    if not uploaded_file:
        st.info("Choose an audio file to enable transcription.")

    if st.button("Transcribe", type="primary", disabled=not uploaded_file):
        _handle_transcription(uploaded_file, prompt, state)

    _render_results(state)


def _handle_transcription(uploaded_file, prompt: str, state: dict) -> None:
    try:
        client = create_openai_client()
    except MissingAPIKeyError as exc:  # pragma: no cover - UI feedback only
        st.error(str(exc))
        return

    status = st.empty()
    progress = st.progress(0.0)
    chunk_details: list[ChunkTranscript] = []
    start_time = time()

    try:
        for chunk in chunked_transcription(
            file=uploaded_file,
            filename=uploaded_file.name,
            prompt=prompt,
            client=client,
            model=DEFAULT_MODEL,
        ):
            chunk_details.append(chunk)
            progress.progress(chunk.progress)
            status.info(
                _format_status_message(
                    chunk_index=chunk.chunk_index,
                    chunk=chunk,
                    start_time=start_time,
                )
            )

    except UnsupportedAudioFormatError as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    progress.progress(1.0)
    status.success(
        f"Completed transcription in {len(chunk_details)} chunk(s) using {DEFAULT_MODEL}."
    )

    transcript_text = "\n\n".join(chunk.text for chunk in chunk_details if chunk.text).strip()

    if not transcript_text:
        st.info("The transcription API did not return any text for this audio.")
        return

    state["transcript"] = transcript_text
    state["error"] = ""

    with st.spinner("Generating LaTeX study notes and spoken-style script..."):
        try:
            documents = generate_latex_documents(transcript_text, client)
        except DocumentGenerationError as exc:
            state["documents"] = []
            state["error"] = str(exc)
            st.error(state["error"])
            return

    state["documents"] = documents


def _render_results(state: dict) -> None:
    transcript = state.get("transcript")
    documents = state.get("documents") or []
    error = state.get("error")

    if error:
        st.error(error)

    if transcript:
        with st.expander("Transcript", expanded=False):
            st.write(transcript)

    if documents:
        _render_documents(documents)


def _render_documents(documents: list[GeneratedDocument]) -> None:
    st.subheader("LaTeX + PDF outputs")

    for doc in documents:
        st.markdown(f"### {doc.title}")
        st.download_button(
            label="Pobierz LaTeX",
            data=doc.latex.encode("utf-8"),
            file_name=doc.latex_filename,
            mime="application/x-tex",
            key=f"download-latex-{doc.key}",
        )
        st.download_button(
            label="Pobierz PDF",
            data=doc.pdf_bytes,
            file_name=doc.pdf_filename,
            mime="application/pdf",
            key=f"download-pdf-{doc.key}",
        )
        with st.expander("Podgląd źródła LaTeX"):
            st.code(doc.latex, language="latex")


def _format_status_message(*, chunk_index: int, chunk: ChunkTranscript, start_time: float) -> str:
    processed_seconds = chunk.end_ms / 1000.0
    total_seconds = chunk.total_ms / 1000.0
    percent = chunk.progress * 100
    eta = _estimate_eta(chunk.progress, start_time)
    eta_display = f"ETA ~{eta}" if eta else "Szacowanie czasu..."
    return (
        f"Chunk {chunk_index + 1}: przetworzono {processed_seconds:.1f}s z {total_seconds:.1f}s "
        f"({percent:.1f}%) | {eta_display}"
    )


def _estimate_eta(progress_fraction: float, start_time: float) -> str | None:
    if progress_fraction <= 0:
        return None
    elapsed = time() - start_time
    if elapsed <= 0:
        return None
    remaining = (elapsed / progress_fraction) - elapsed
    if remaining < 0:
        remaining = 0
    return _format_duration(remaining)


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main() -> None:
    """Convenience runner for `streamlit run`."""

    build_streamlit_app()


if __name__ == "__main__":
    main()

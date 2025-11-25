"""Audio chunking + transcription helpers for the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Optional

from openai import OpenAI
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from vi_speech_to_text.openai_client import create_openai_client

# API limits uploads to 25 MB so stay below that for safety.
_MAX_CHUNK_BYTES = 24 * 1024 * 1024
_MIN_CHUNK_MS = 5_000  # never split into segments shorter than 5 seconds unless needed.
_SUPPORTED_EXTENSIONS = {
    "mp3",
    "mp4",
    "mpeg",
    "mpga",
    "m4a",
    "wav",
    "webm",
}

_EXPORT_FORMAT_OVERRIDES = {
    # Map file extensions to the FFmpeg format names that successfully round trip.
    "m4a": "mp4",
    "mpga": "mp3",
    "mpeg": "mp3",
}


class UnsupportedAudioFormatError(ValueError):
    """Raised when a provided audio file has an unsupported extension."""


@dataclass
class ChunkTranscript:
    """Represents the transcription result for a single chunk."""

    chunk_index: int
    start_ms: int
    end_ms: int
    total_ms: int
    text: str

    @property
    def progress(self) -> float:
        """Fraction of the file processed when this chunk completes."""

        return min(self.end_ms / self.total_ms, 1.0) if self.total_ms else 0.0


def supported_audio_extensions() -> Iterable[str]:
    """Return the audio extensions the UI can ingest."""

    return tuple(sorted(_SUPPORTED_EXTENSIONS))


def chunked_transcription(
    *,
    file: BinaryIO,
    filename: str,
    prompt: str = "",
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-transcribe",
    max_chunk_bytes: int = _MAX_CHUNK_BYTES,
) -> Iterator[ChunkTranscript]:
    """Transcribe an uploaded file chunk-by-chunk."""

    audio_format = _infer_audio_format(filename)
    audio_segment = _load_audio_segment(file, audio_format)
    total_ms = len(audio_segment)

    api_client = client or create_openai_client()
    prompt_value = prompt.strip() or None

    for chunk_index, start_ms, end_ms, payload in _generate_chunks(
        audio_segment=audio_segment,
        audio_format=audio_format,
        max_chunk_bytes=max_chunk_bytes,
    ):
        response = api_client.audio.transcriptions.create(
            model=model,
            file=payload,
            response_format="text",
            prompt=prompt_value,
        )
        text = _extract_transcript_text(response)
        yield ChunkTranscript(
            chunk_index=chunk_index,
            start_ms=start_ms,
            end_ms=end_ms,
            total_ms=total_ms,
            text=text,
        )


def _infer_audio_format(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix in _SUPPORTED_EXTENSIONS:
        return suffix
    raise UnsupportedAudioFormatError(
        "Unsupported audio type. Please upload one of: "
        + ", ".join(sorted(_SUPPORTED_EXTENSIONS))
    )


def _load_audio_segment(file: BinaryIO, audio_format: str) -> AudioSegment:
    file.seek(0)
    try:
        return AudioSegment.from_file(file, format=audio_format)
    except CouldntDecodeError as exc:  # pragma: no cover - depends on external ffmpeg
        raise ValueError("Unable to decode the provided audio file.") from exc


def _generate_chunks(
    *, audio_segment: AudioSegment, audio_format: str, max_chunk_bytes: int
) -> Iterator[tuple[int, int, int, BytesIO]]:
    total_ms = len(audio_segment)
    approx_chunk_ms = _estimate_chunk_duration(audio_segment, max_chunk_bytes)
    export_format, chunk_extension = _resolve_export_settings(audio_format)

    start_ms = 0
    chunk_index = 0
    while start_ms < total_ms:
        remaining_ms = total_ms - start_ms
        target_duration = min(approx_chunk_ms, remaining_ms)

        while True:
            chunk = audio_segment[start_ms : start_ms + target_duration]
            payload = BytesIO()
            chunk.export(payload, format=export_format)
            size = payload.tell()

            if size <= max_chunk_bytes:
                payload.seek(0)
                payload.name = f"chunk-{chunk_index}.{chunk_extension}"
                yield chunk_index, start_ms, start_ms + target_duration, payload
                start_ms += target_duration
                chunk_index += 1
                break

            if target_duration <= _MIN_CHUNK_MS:
                raise ValueError(
                    "Chunk remained above the upload limit even at the minimum length."
                )

            target_duration = max(_MIN_CHUNK_MS, int(target_duration * 0.7))


def _estimate_chunk_duration(audio_segment: AudioSegment, max_chunk_bytes: int) -> int:
    # Bytes per millisecond (frame_rate * frame_width describes bytes per millisecond * 1000).
    bytes_per_second = audio_segment.frame_rate * audio_segment.frame_width
    bytes_per_ms = max(bytes_per_second / 1000.0, 1)
    approx_ms = int(max_chunk_bytes / bytes_per_ms)
    return max(approx_ms, _MIN_CHUNK_MS)


def _resolve_export_settings(audio_format: str) -> tuple[str, str]:
    """Return the ffmpeg export format and extension to use for chunks."""

    export_format = _EXPORT_FORMAT_OVERRIDES.get(audio_format, audio_format)
    if export_format == audio_format:
        return audio_format, audio_format
    return export_format, export_format


def _extract_transcript_text(response: object) -> str:
    """Normalize transcription responses to raw text."""

    if isinstance(response, str):
        return response.strip()
    text = getattr(response, "text", "")
    return (text or "").strip()

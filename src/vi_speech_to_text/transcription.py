"""Audio chunking + transcription helpers for the Streamlit UI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Optional

from openai import OpenAI

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
    "ogg",
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


@dataclass
class _PreparedAudioFile:
    path: str
    size_bytes: int


@dataclass
class _AudioMetadata:
    duration_ms: int
    bit_rate: Optional[int]


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
    api_client = client or create_openai_client()
    prompt_value = prompt.strip() or None

    with _prepare_local_audio_file(file) as prepared_file:
        metadata = _probe_audio_metadata(prepared_file.path)
        total_ms = metadata.duration_ms
        approx_chunk_ms = _estimate_chunk_duration(
            duration_ms=metadata.duration_ms,
            bit_rate=metadata.bit_rate,
            file_size=prepared_file.size_bytes,
            max_chunk_bytes=max_chunk_bytes,
        )

        for chunk_index, start_ms, end_ms, payload in _generate_chunks(
            input_path=prepared_file.path,
            audio_format=audio_format,
            approx_chunk_ms=approx_chunk_ms,
            total_ms=total_ms,
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


@contextmanager
def _prepare_local_audio_file(file: BinaryIO) -> Iterator[_PreparedAudioFile]:
    """Ensure the audio upload exists on disk for ffmpeg/ffprobe consumption."""

    file.seek(0)
    source_path = getattr(file, "name", None)
    if isinstance(source_path, str) and os.path.isfile(source_path):
        size_bytes = os.path.getsize(source_path)
        yield _PreparedAudioFile(path=source_path, size_bytes=size_bytes)
        return

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        shutil.copyfileobj(file, tmp_file)
        tmp_path = tmp_file.name

    size_bytes = os.path.getsize(tmp_path)
    try:
        yield _PreparedAudioFile(path=tmp_path, size_bytes=size_bytes)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def _probe_audio_metadata(path: str) -> _AudioMetadata:
    """Return quick metadata for the provided audio file via ffprobe."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=duration,bit_rate",
        "-of",
        "json",
        path,
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on system ffmpeg
        raise RuntimeError("ffprobe is required but was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - ffprobe failure
        raise ValueError("Unable to inspect the provided audio file.") from exc

    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - unexpected ffprobe output
        raise ValueError("Received malformed metadata from ffprobe.") from exc

    duration_seconds = _extract_duration_seconds(data)
    if duration_seconds is None or duration_seconds <= 0:
        raise ValueError("Unable to determine audio duration from the provided file.")
    bit_rate = _extract_bit_rate(data)

    duration_ms = max(int(duration_seconds * 1000), 1)
    return _AudioMetadata(duration_ms=duration_ms, bit_rate=bit_rate)


def _extract_duration_seconds(data: dict) -> Optional[float]:
    sources = []
    if isinstance(data.get("format"), dict):
        sources.append(data["format"].get("duration"))
    if isinstance(data.get("streams"), list):
        for stream in data["streams"]:
            if isinstance(stream, dict):
                sources.append(stream.get("duration"))

    for value in sources:
        duration = _safe_float(value)
        if duration:
            return duration
    return None


def _extract_bit_rate(data: dict) -> Optional[int]:
    candidates = []
    if isinstance(data.get("format"), dict):
        candidates.append(data["format"].get("bit_rate"))
    if isinstance(data.get("streams"), list):
        for stream in data["streams"]:
            if isinstance(stream, dict):
                candidates.append(stream.get("bit_rate"))

    for value in candidates:
        bit_rate = _safe_int(value)
        if bit_rate:
            return bit_rate
    return None


def _export_chunk(
    *,
    input_path: str,
    export_format: str,
    chunk_extension: str,
    start_ms: int,
    duration_ms: int,
) -> BytesIO:
    """Use ffmpeg to export a time-bounded slice of the source file."""

    start_seconds = max(start_ms, 0) / 1000.0
    duration_seconds = max(duration_ms, 1) / 1000.0
    output_flags: list[str] = []
    if export_format in {"mp4"}:
        output_flags.extend(["-movflags", "frag_keyframe+empty_moov"])

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        input_path,
    ]
    cmd.extend(output_flags)
    cmd.extend(["-vn"])

    tmp_file = tempfile.NamedTemporaryFile(suffix=f".{chunk_extension}", delete=False)
    tmp_file.close()
    cmd.extend(["-f", export_format, "-y", tmp_file.name])
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on system ffmpeg
        raise RuntimeError("ffmpeg is required but was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - ffmpeg failure
        stderr_text = (exc.stderr or b"").decode("utf-8", errors="ignore").strip()
        raise ValueError(
            "Unable to export audio chunk via ffmpeg." + (f" {stderr_text}" if stderr_text else "")
        ) from exc

    try:
        with open(tmp_file.name, "rb") as tmp_in:
            payload = BytesIO(tmp_in.read())
    finally:
        try:
            os.remove(tmp_file.name)
        except FileNotFoundError:
            pass

    payload.seek(0)
    return payload


def _generate_chunks(
    *,
    input_path: str,
    audio_format: str,
    approx_chunk_ms: int,
    total_ms: int,
    max_chunk_bytes: int,
) -> Iterator[tuple[int, int, int, BytesIO]]:
    export_format, chunk_extension = _resolve_export_settings(audio_format)

    start_ms = 0
    chunk_index = 0
    while start_ms < total_ms:
        remaining_ms = total_ms - start_ms
        target_duration = min(approx_chunk_ms, remaining_ms)

        while True:
            payload = _export_chunk(
                input_path=input_path,
                export_format=export_format,
                chunk_extension=chunk_extension,
                start_ms=start_ms,
                duration_ms=target_duration,
            )
            size = payload.getbuffer().nbytes

            if size <= max_chunk_bytes:
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


def _estimate_chunk_duration(
    *,
    duration_ms: int,
    bit_rate: Optional[int],
    file_size: int,
    max_chunk_bytes: int,
) -> int:
    bytes_per_ms: Optional[float] = None
    if bit_rate:
        bytes_per_ms = max((bit_rate / 8) / 1000.0, 1.0)
    elif duration_ms:
        bytes_per_ms = max(file_size / max(duration_ms, 1), 1.0)

    approx_ms = int(max_chunk_bytes / bytes_per_ms) if bytes_per_ms else max_chunk_bytes
    return max(approx_ms, _MIN_CHUNK_MS)


def _resolve_export_settings(audio_format: str) -> tuple[str, str]:
    """Return the ffmpeg export format and extension to use for chunks."""

    export_format = _EXPORT_FORMAT_OVERRIDES.get(audio_format, audio_format)
    if export_format == audio_format:
        return audio_format, audio_format
    return export_format, export_format


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_transcript_text(response: object) -> str:
    """Normalize transcription responses to raw text."""

    if isinstance(response, str):
        return response.strip()
    text = getattr(response, "text", "")
    return (text or "").strip()

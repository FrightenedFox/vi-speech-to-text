# VI Speech-to-Text

A Streamlit application that turns long-form audio into text and two polished LaTeX documents:

1. Exam-ready study notes produced by `gpt-4o-transcribe` + `gpt-4.1`.
2. A “spoken-style” script that keeps the lecturer’s voice while cleaning filler language.

Each LaTeX file is compiled locally into a PDF via `pdflatex`, and both the source and PDF can be downloaded from the UI.

## Feature Highlights

- Upload large audio files (mp3, m4a, wav, webm, mpga, etc.); the app auto-chunks them below the 25 MB OpenAI limit.
- Optional prompt field to bias the transcription (speaker names, jargon, context).
- Progress bar with percentage + ETA while `gpt-4o-transcribe` processes the audio.
- Transcript stored in session state and shown inside an expander so the page never resets when downloading artifacts.
- Parallel `gpt-4.1` calls generate two LaTeX variants, each compiled to PDF.

## Prerequisites

You need the following regardless of operating system:

- **OpenAI account + API key** (set via `.env`, details below).
- **Git** for cloning the repository.
- **Python 3.11** (Poetry manages a virtualenv for dependencies).
- **Poetry** for dependency/runtime management.
- **FFmpeg** so `pydub` can load and chunk audio files.
- **A LaTeX distribution** that provides the `pdflatex` command (e.g., TeX Live, MiKTeX).

If you do not currently have Python or these tools installed, follow the platform-specific guides below.

## Windows Setup (fresh machine)

1. **Install Git**
   - Download the latest installer from <https://git-scm.com/download/win> and accept the defaults (ensure “Git from the command line” is selected).

2. **Install Python 3.11**
   - Grab the Windows installer from <https://www.python.org/downloads/windows/>.
   - During installation, check **“Add python.exe to PATH”**.

3. **Install Poetry**
   - Open PowerShell (Win+X → Windows Terminal) and run:
     ```powershell
     (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
     ```
   - Restart the terminal so `poetry` is on PATH.

4. **Install FFmpeg**
   - Easiest path is [Chocolatey](https://chocolatey.org/) (optional), but if you prefer manual:
     - Download a static build from <https://www.gyan.dev/ffmpeg/builds/>.
     - Extract it (e.g., `C:\ffmpeg`) and add `C:\ffmpeg\bin` to your PATH environment variable.

5. **Install LaTeX**
   - Install [MiKTeX](https://miktex.org/download) or [TeX Live](https://www.tug.org/texlive/acquire-netinstall.html).
   - After installation, open a new terminal and verify `pdflatex --version` prints version info.

6. **Verify toolchain**
   ```powershell
   git --version
   python --version
   poetry --version
   ffmpeg -version
   pdflatex --version
   ```

## Linux Setup (Ubuntu/Debian example)

1. Install required packages via APT:
   ```bash
   sudo apt update
   sudo apt install -y git python3 python3-venv python3-pip ffmpeg texlive-latex-extra texlive-lang-polish
   ```
   > The `texlive` packages include `pdflatex`. Adjust for your distribution (e.g., `dnf` on Fedora).

2. Install Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   Reload your shell (`source ~/.profile` or open a new terminal) so `poetry` is available.

3. Confirm tools exist:
   ```bash
   git --version
   python3 --version
   poetry --version
   ffmpeg -version
   pdflatex --version
   ```

## Project Setup (all platforms)

1. **Clone the repository**
   ```bash
   git clone https://github.com/FrightenedFox/vi-speech-to-text.git
   cd vi-speech-to-text
   ```

2. **Configure environment variables**
   ```bash
   cp example.env .env
   # Edit .env in your editor and set OPENAI_API_KEY=sk-...
   ```
   Additional env vars (`OPENAI_ORG_ID`, `OPENAI_BASE_URL`) can also be set if needed.

3. **Install dependencies**
   ```bash
   poetry install
   ```
   Poetry creates an isolated virtualenv and installs Streamlit, FFmpeg bindings, OpenAI SDK, etc.

4. **Run the Streamlit UI**
   ```bash
   poetry run streamlit run streamlit-frontend/app.py
   ```

5. **Use the app**
   - Upload an audio file (mp3, m4a, wav, webm, mpga, mpeg).
   - Optionally provide a transcription prompt.
   - Watch the progress bar (percent + ETA). When done, expand the transcript, then download the LaTeX/PDF outputs for both study notes and the spoken-style script.

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `ffmpeg` not found | Ensure `ffmpeg -version` works in the same terminal before running Streamlit. Add it to PATH if necessary. |
| `pdflatex` missing | Install a LaTeX distribution and restart the terminal. `pdflatex --version` should succeed. |
| `OPENAI_API_KEY` errors | Confirm `.env` exists and contains `OPENAI_API_KEY`. Streamlit logs will show a warning if the key is missing. |
| Poetry not recognized | Ensure the Poetry installer directory (e.g., `%APPDATA%\pypoetry\venv\Scripts` on Windows or `$HOME/.local/bin` on Linux) is on PATH. |

## Repository Layout

```
.
├── src/vi_speech_to_text/
│   ├── app.py               # Streamlit UI + session state
│   ├── transcription.py     # Audio chunking + gpt-4o-transcribe integration
│   ├── postprocess.py       # GPT-4.1 LaTeX generation + pdflatex compilation
│   └── openai_client.py     # Client factory + dotenv loading
├── streamlit-frontend/
│   └── app.py               # Thin launcher that imports the package entrypoint
├── pyproject.toml           # Poetry metadata
├── poetry.lock              # Locked dependencies
└── README.md                # This guide
```

You’re ready to explore transcription-to-LaTeX pipelines on both Windows and Linux—even on machines that didn’t have Python installed beforehand. Happy experimenting!

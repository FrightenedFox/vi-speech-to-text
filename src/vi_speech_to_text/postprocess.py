"""Convert transcripts into LaTeX outputs and compile PDFs."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Iterable, List, Sequence

from openai import OpenAI

STUDY_NOTES_PROMPT = """You are a language model that converts lecture transcripts into well-formatted LaTeX study notes that can be compiled into a clean, readable PDF.

--------------------------------
TASK DESCRIPTION (READ CAREFULLY)
--------------------------------

You will receive:

- A raw transcript of a university lecture (e.g. on the history of Italian literature), written in Polish.
- The transcript will be messy: spoken language, repetitions, false starts, small mistakes, and no structure.

Your job:

1. **Convert the transcript into a SINGLE LaTeX document** that:
   - Can be compiled to PDF without modification.
   - Is clearly structured and pleasant to read.
   - Is suitable as a study script to prepare for an exam.
   - Preserves *all* the meaningful information from the lecture (names, dates, terms, distinctions, examples, historical context, definitions, etc.).

2. **Structure and clarity:**
   - Organize the material into a small number of logical sections and subsections (e.g., for different authors, periods, genres, concepts).
   - Make the text coherent: correct obvious transcription errors, fix grammar and punctuation, and join broken sentences.
   - You may paraphrase for clarity, but **do not omit important content.** If something is repeated in the transcript, you may:
     - merge repetitions into one clear explanation, or
     - keep both if they add nuance.
   - You can insert short clarifying phrases or definitions where helpful, but do **not** invent new facts.

3. **Use fewer lists than in the example:**
   - Use **lists (`itemize` / `enumerate`) only when they genuinely improve readability** (for enumerating key points, characteristics, or examples).
   - Prefer **normal paragraphs** over long or nested lists.
   - The document should read like a well-structured script or set of lecture notes, not like a bullet-point slide deck.

4. **Keep and highlight key academic details:**
   - Preserve:
     - Names of authors and works,
     - Titles (in original language; use `\emph{...}`),
     - Historical periods, dates, battles, political factions (gwelfowie / gibelini),
     - Definitions of genres (e.g. lauda liryczna / dramatyczna, ballata, sonet),
     - Main theses and interpretations from the lecture.
   - Use section titles/subtitles to make it easy to locate topics (e.g. “Lauda liryczna i dramatyczna”, “Jacopone da Todi”, “Guittone d’Arezzo”, “Dolce Stil Nuovo”).

5. **LaTeX requirements:**
   - You must output a **complete, compilable LaTeX document**, starting with `\documentclass` and ending with `\end{document}`.
   - Use the template given below as a base and adapt it:
     - Keep the general preamble, but you may add additional standard packages if necessary.
     - The hyperref configuration must not produce colored boxes (use `hidelinks`).
   - Use `\section`, `\subsection`, `\subsubsection` where needed, but do **not** over-segment the text into too many tiny sections.
   - Use Polish language conventions (Polish text, Polish diacritics, Polish typographic norms as much as LaTeX allows).

6. **Output format requirement (IMPORTANT):**
   - Your response must contain **only the LaTeX source code** of the final document.
   - Do **NOT** include any explanation, comments to the user, backticks, or markdown code fences.
   - Do **NOT** include placeholders like “TODO” or “here goes the text”.
   - Just output the final `.tex` content.

--------------------------------
LATEX TEMPLATE TO USE AND ADAPT
--------------------------------

Use this template. Insert your reworked lecture content in the place marked `%%% CONTENT STARTS HERE %%%`. You may rename sections and adjust structure as needed.

\\documentclass[12pt,a4paper]{article}

\\usepackage[polish]{babel}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}

\\usepackage{geometry}
\\geometry{margin=2.5cm}

\\usepackage{setspace}
\\onehalfspacing

\\usepackage{microtype}

\\usepackage{hyperref}
\\hypersetup{
    hidelinks,
    pdfauthor={Notatki z wykładu},
    pdftitle={Historia literatury włoskiej -- notatki z wykładu},
    pdfcreator={LaTeX}
}

\\usepackage{enumitem}
\\setlist{nosep}

\\usepackage{csquotes}

\\usepackage{titlesec}
\\titleformat{\\section}{\\normalfont\\Large\\bfseries}{\\thesection.}{0.6em}{}
\\titleformat{\\subsection}{\\normalfont\\large\\bfseries}{\\thesubsection.}{0.5em}{}
\\titleformat{\\subsubsection}{\\normalfont\\normalsize\\bfseries}{\\thesubsubsection.}{0.4em}{}

\\begin{document}

\\title{Historia literatury włoskiej\\\\Notatki z wykładu}
\\author{Na podstawie transkrypcji wykładu uniwersyteckiego}
\\date{}
\\maketitle

\\tableofcontents
\\newpage

%%% CONTENT STARTS HERE %%%

% Przekształć poniższą transkrypcję w spójny, dobrze zredagowany tekst,
% używając sekcji, podsekcji i tylko umiarkowanej liczby list.

% Tutaj wstaw swój zredagowany tekst na podstawie transkrypcji:
% (Zastąp ten komentarz finalną wersją treści.)

%%% CONTENT ENDS HERE %%%

\\end{document}

--------------------------------
INPUT FORMAT
--------------------------------

Po tym poleceniu otrzymasz transkrypcję wykładu umieszczoną pomiędzy znacznikami:

BEGIN_TRANSCRIPT
...
END_TRANSCRIPT

Twoje zadanie: na podstawie tej transkrypcji wygenerować kompletny dokument LaTeX zgodnie z powyższymi wytycznymi, używając podanego szablonu.

--------------------------------
FINAL REMINDER
--------------------------------

W odpowiedzi podaj **wyłącznie** kompletny kod LaTeX dokumentu, bez komentarzy, bez opisu zadania, bez znaczników „BEGIN_TRANSCRIPT/END_TRANSCRIPT” i bez znaczników markdown (np. ```latex).

Then, after producing the LaTeX document, return only that LaTeX. Do not add explanations.
"""

SPOKEN_STYLE_PROMPT = """You are a language model that converts lecture transcripts into clean, lightly edited LaTeX “spoken-style” scripts that can be compiled into a PDF.

Unlike a summary, your goal is to preserve the lecturer’s voice and wording as much as possible, but without typical spoken disfluencies.

--------------------------------
TASK DESCRIPTION (READ CAREFULLY)
--------------------------------

You will receive:

- A raw transcript of a university lecture (in Polish), as it was spoken.
- It will contain:
  - powtórzenia (repetitions),
  - wtrącenia typu „no, prawda, jakby…”, „yyy”, „yyy no, tak”,
  - przerwane zdania i poprawki w locie,
  - drobne błędy składniowe i stylistyczne typowe dla mowy.

Your job:

1. **Produce a SINGLE LaTeX document** that:
   - Can be compiled directly to PDF.
   - Zawiera tekst jak najbardziej zbliżony do oryginalnej wypowiedzi.
   - Brzmi jak płynny, mówiony wykład (tak jakby ktoś go wygłaszał na żywo), ale:
     - bez „yyy”, „no więc”, „tak, tak”, zbędnych powtórzeń,
     - bez niegramatycznych przerw i porzuconych zdań,
     - bez typowych ustnych „haczeń”.

2. **Zachowaj możliwie wierne brzmienie wykładu (word-to-word sens):**
   - Nie streszczaj, nie skracaj treści merytorycznej:
     - Nie opuszczaj informacji, nazwisk, dat, pojęć, przykładów, dygresji, które niosą sens.
   - Możesz:
     - łączyć urwane zdania w jedno poprawne,
     - poprawiać szyk zdania tak, by brzmiało naturalnie w języku pisanym, ale nadal „mówionym”,
     - usuwać powtórzenia typu: „lauda, lauda, no lauda, prawda” → „lauda, prawda”.
   - Nie zmieniaj znaczenia wypowiedzi.
   - Nie dodawaj nowych treści ani interpretacji, które nie wynikają wprost z oryginału.

3. **Styl i struktura:**
   - Dokument ma wyglądać jak:
     - zredagowany zapis mówionego wykładu,
     - z wyraźnymi **sekcjami i podsekcjami** odzwierciedlającymi zmiany tematu.
   - Możesz:
     - dodawać nagłówki `\\section` i `\\subsection` przy wyraźnych przejściach tematycznych (np. „Lauda”, „Jacopone da Todi”, „Guittone d’Arezzo”, „Poezja toskańska”, „Dolce Stil Nuovo”),
     - dzielić tekst na akapity tam, gdzie w mowie występują „bloki” myśli.
   - Utrzymaj charakter mówiony:
     - możesz zachować zwroty do studentów („jak państwo widzą”, „będziemy o tym mówić za chwilę”),
     - możesz zachować lekką powtarzalność, jeśli jest typowa dla stylu wykładu,
     - ale usuwaj czyste „szumy” językowe.

4. **Jak traktować typowe „ludzkie” błędy:**
   - USUWAJ lub poprawiaj:
     - „yyy”, „ee”, „no tak”, „prawda?”, używane wyłącznie jako wypełniacze,
     - oczywiste przejęzyczenia (np. „Hohenzaltów” → „Hohenstaufów”, jeśli jasno wynika z kontekstu),
     - powtórzenia całych fraz tylko po to, by się poprawić (weź ostatnią, poprawną wersję).
   - ZOSTAW:
     - żartobliwe uwagi o „jak już państwo wiedzą”,
     - odniesienia do „przeczytamy to w przyszłym roku”, „będziemy mieć ten tekst na zajęciach” itp.,
       bo są częścią naturalnego stylu wykładu.

5. **Używaj sekcji, ale z umiarem:**
   - Stosuj `\\section`, `\\subsection`, ewentualnie `\\subsubsection` tam, gdzie wykładowca wyraźnie przechodzi do nowego bloku tematycznego.
   - Nie dziel tekstu na zbyt wiele małych sekcji – tekst ma płynąć jak wykład.
   - Możesz korzystać z list (`itemize`, `enumerate`) tylko tam, gdzie mówca wylicza elementy (np. cechy gatunku, nazwy autorów), ale nie rób z dokumentu „bullet-pointowego” konspektu.

6. **Zachowaj wszystkie dane merytoryczne:**
   - Nie wolno Ci usuwać:
     - nazw autorów,
     - tytułów dzieł (zapisuj je w `\\emph{...}`),
     - dat, wydarzeń historycznych, nazw stronnictw (gwelfowie, gibelini),
     - głównych tez interpretacyjnych i definicji gatunków.

7. **Wymogi techniczne LaTeX:**
   - Musisz zwrócić **kompletny, kompilowalny dokument LaTeX**:
     - zaczynający się od `\\documentclass`,
     - kończący się na `\\end{document}`.
   - Użyj podanego poniżej szablonu i dopasuj go:
     - zachowaj preambułę,
     - możesz dodać standardowe pakiety, jeśli naprawdę są potrzebne,
     - hiperłącza mają nie generować kolorowych ramek (użyj `hidelinks`).

8. **Format odpowiedzi (KLUCZOWE):**
   - Odpowiadasz **wyłącznie** kodem LaTeX gotowego dokumentu.
   - Nie dodawaj żadnych wyjaśnień, komentarzy, tekstu w Markdown, ani znaczników typu ```latex.
   - Nie używaj komentarzy typu `% TODO` wewnątrz treści wykładu.
   - Tylko końcowy `.tex`.

--------------------------------
LATEX TEMPLATE TO USE AND ADAPT
--------------------------------

Użyj tego szablonu. W miejsce oznaczone `%%% CONTENT STARTS HERE %%%` wstaw przetworzony, płynny tekst wykładu (z sekcjami i akapitami). Możesz zmienić tytuł dokumentu, sekcje itp., ale zachowaj ogólną strukturę.

\\documentclass[12pt,a4paper]{article}

\\usepackage[polish]{babel}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}

\\usepackage{geometry}
\\geometry{margin=2.5cm}

\\usepackage{setspace}
\\onehalfspacing

\\usepackage{microtype}

\\usepackage{hyperref}
\\hypersetup{
    hidelinks,
    pdfauthor={Notatki z wykładu},
    pdftitle={Historia literatury włoskiej -- zapis wykładu},
    pdfcreator={LaTeX}
}

\\usepackage{enumitem}
\\setlist{nosep}

\\usepackage{csquotes}

\\usepackage{titlesec}
\\titleformat{\\section}{\\normalfont\\Large\\bfseries}{\\thesection.}{0.6em}{}
\\titleformat{\\subsection}{\\normalfont\\large\\bfseries}{\\thesubsection.}{0.5em}{}
\\titleformat{\\subsubsection}{\\normalfont\\normalsize\\bfseries}{\\thesubsubsection.}{0.4em}{}

\\begin{document}

\\title{Historia literatury włoskiej\\\\Zapis mówionego wykładu}
\\author{Na podstawie transkrypcji wykładu uniwersyteckiego}
\\date{}
\\maketitle

\\tableofcontents
\\newpage

%%% CONTENT STARTS HERE %%%

% Tutaj wstaw płynny, poprawiony zapis wykładu
% na podstawie dostarczonej transkrypcji.

%%% CONTENT ENDS HERE %%%

\\end{document}

--------------------------------
INPUT FORMAT
--------------------------------

Po tym poleceniu otrzymasz transkrypcję wykładu umieszczoną pomiędzy znacznikami:

BEGIN_TRANSCRIPT
...
END_TRANSCRIPT

Twoje zadanie: na podstawie tej transkrypcji wygenerować kompletny dokument LaTeX zgodnie z powyższymi wytycznymi, używając podanego szablonu, zachowując możliwie wiernie brzmienie wykładu.

--------------------------------
FINAL REMINDER
--------------------------------

W odpowiedzi podaj **wyłącznie** kompletny kod LaTeX dokumentu, bez komentarzy, bez opisu zadania, bez znaczników „BEGIN_TRANSCRIPT/END_TRANSCRIPT” i bez znaczników markdown (np. ```latex).

Then, after producing the LaTeX document, return only that LaTeX. Do not add explanations.
"""


@dataclass
class GeneratedDocument:
    """Represents a LaTeX artifact plus compiled PDF."""

    key: str
    title: str
    latex: str
    pdf_bytes: bytes
    latex_filename: str
    pdf_filename: str


class DocumentGenerationError(RuntimeError):
    """Raised when GPT or PDF generation fails."""


def generate_latex_documents(transcript: str, client: OpenAI) -> List[GeneratedDocument]:
    """Create both LaTeX variants and compile them to PDFs in parallel."""

    tasks: Sequence[tuple[str, str, str]] = (
        ("study-notes", "LaTeX – notatki", STUDY_NOTES_PROMPT),
        ("spoken-script", "LaTeX – zapis mówiony", SPOKEN_STYLE_PROMPT),
    )

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {
            executor.submit(_generate_single_document, key, title, prompt, transcript, client): key
            for key, title, prompt in tasks
        }

        documents: list[GeneratedDocument] = []
        errors: list[tuple[str, Exception]] = []
        for future in as_completed(futures):
            key = futures[future]
            try:
                documents.append(future.result())
            except Exception as exc:  # pragma: no cover - surfaced to UI
                errors.append((key, exc))

    if errors:
        error_messages = "; ".join(f"{key}: {exc}" for key, exc in errors)
        raise DocumentGenerationError(
            "Failed to generate all LaTeX documents: " + error_messages
        )

    # Preserve original ordering from tasks tuple.
    ordering = {key: index for index, (key, *_rest) in enumerate(tasks)}
    documents.sort(key=lambda doc: ordering[doc.key])
    return documents


def _generate_single_document(
    key: str, title: str, prompt: str, transcript: str, client: OpenAI
) -> GeneratedDocument:
    latex = _call_gpt_latex(prompt, transcript, client)
    pdf_bytes = _compile_pdf(latex)
    return GeneratedDocument(
        key=key,
        title=title,
        latex=latex,
        pdf_bytes=pdf_bytes,
        latex_filename=f"{key}.tex",
        pdf_filename=f"{key}.pdf",
    )


def _call_gpt_latex(prompt: str, transcript: str, client: OpenAI) -> str:
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"BEGIN_TRANSCRIPT\n{transcript}\nEND_TRANSCRIPT",
            },
        ],
    )
    text = _extract_response_text(response)
    if not text.strip():
        raise DocumentGenerationError("GPT returned an empty LaTeX document.")
    return text


def _extract_response_text(response: object) -> str:
    """Best-effort extraction of text from OpenAI response objects."""

    if isinstance(response, str):
        return response

    for attr in ("output_text", "text"):
        value = getattr(response, attr, None)
        if isinstance(value, list):
            return "".join(str(part) for part in value)
        if isinstance(value, str):
            return value

    output = getattr(response, "output", None)
    if output:
        texts: list[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for block in content:
                text_value = getattr(block, "text", None) or getattr(block, "value", None)
                if text_value:
                    texts.append(str(text_value))
        if texts:
            return "".join(texts)

    return str(response)


def _compile_pdf(latex: str) -> bytes:
    with TemporaryDirectory() as tmpdir:
        tex_path = f"{tmpdir}/document.tex"
        with open(tex_path, "w", encoding="utf-8") as handle:
            handle.write(latex)

        for run in range(2):
            proc = subprocess.run(
                ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", "document.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise DocumentGenerationError(
                    "pdflatex failed: " + (proc.stderr or proc.stdout or "unknown error")
                )
            # First pass generates TOC/aux data; continue to second pass automatically.

        pdf_path = f"{tmpdir}/document.pdf"
        try:
            with open(pdf_path, "rb") as pdf_handle:
                return pdf_handle.read()
        except FileNotFoundError as exc:  # pragma: no cover
            raise DocumentGenerationError("pdflatex did not produce a PDF file.") from exc

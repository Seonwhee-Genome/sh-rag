"""Extract plain text from supported document formats."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".log", ".ini", ".toml", ".xml",
    ".py", ".go", ".js", ".ts", ".java", ".kt", ".c", ".cpp", ".h", ".rs", ".sql", ".sh", ".ps1",
}
SUPPORTED_EXTS = TEXT_EXTS | {".pdf", ".docx", ".html", ".htm"}


@dataclass
class Section:
    """A span of document text, optionally tied to a 1-based page number."""

    text: str
    page: int | None = None


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS and not path.name.startswith((".", "~$"))


def load(path: Path) -> list[Section]:
    """Return the non-empty text sections of a document."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        sections = _load_pdf(path)
    elif ext == ".docx":
        sections = _load_docx(path)
    elif ext in {".html", ".htm"}:
        sections = [Section(_html_to_text(read_text(path)))]
    elif ext in TEXT_EXTS:
        sections = [Section(read_text(path))]
    else:
        raise ValueError(f"unsupported file type: {ext}")
    cleaned = [Section(normalize(s.text), s.page) for s in sections]
    return [s for s in cleaned if s.text]


def read_text(path: Path) -> str:
    """Decode a text file, trying UTF-8 first, then CP949 (common for Korean Windows files)."""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [Section(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]


def _load_docx(path: Path) -> list[Section]:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return [Section("\n\n".join(parts))]


def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return soup.get_text("\n")

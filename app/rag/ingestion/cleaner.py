"""Text cleaning applied to raw PDF-extracted text before chunking."""
import re

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize whitespace/artifacts from PDF extraction.

    Does not touch wording or casing — only removes null bytes and
    collapses redundant whitespace, so chunk content still matches what
    a human would read on the page.
    """
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_BLANK_LINE.sub("\n\n", text)
    return text.strip()

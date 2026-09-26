"""Run the bundled DOCX renderer with the installed Windows LibreOffice."""

from __future__ import annotations

import runpy
import shutil
import sys


RENDER = (
    r"C:\Users\Administrator\.codex\plugins\cache\openai-primary-runtime"
    r"\documents\26.921.10847\skills\documents\render_docx.py"
)
SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"
_which = shutil.which


def which(program, *args, **kwargs):
    if program == "soffice.exe":
        return SOFFICE
    return _which(program, *args, **kwargs)


shutil.which = which
sys.argv[0] = RENDER
runpy.run_path(RENDER, run_name="__main__")

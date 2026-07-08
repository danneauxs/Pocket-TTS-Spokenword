@echo off
REM DocDNA setup — install Python dependencies for Mr. Smith
REM Run this once on a new machine to enable incremental docstring syncing.
REM Querying (docdna_query.py) works without this — no setup needed.

cd /d "%~dp0"
echo Setting up DocDNA environment...
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt --quiet
echo Done. Mr. Smith is ready.
echo Run:  python docdna_sync.py ^<project^> --files ^<changed_files^>

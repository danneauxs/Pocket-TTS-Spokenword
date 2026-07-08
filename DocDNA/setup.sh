#!/bin/sh
# DocDNA setup — install Python dependencies for Mr. Smith
# Run this once on a new machine to enable incremental docstring syncing.
# Querying (docdna_query.py) works without this — no setup needed.

cd "$(dirname "$0")"
echo "Setting up DocDNA environment..."
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --quiet
echo "Done. Mr. Smith is ready."
echo "Run:  python3 docdna_sync.py <project> --files <changed_files>"

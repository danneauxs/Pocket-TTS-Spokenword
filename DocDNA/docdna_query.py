#!/usr/bin/env python3
"""
docdna-query — CLI tool for querying DocDNA project knowledge databases.

Usage:
    docdna-query /path/to/project [--subcommand] [args...]

The DocDNA folder is auto-detected as <project>/DocDNA/.
All subcommands read from the SQLite database (docdna.db) only.
JSON and Markdown files in the DocDNA directory are for human readers only.

Output format: markdown (default), --json for structured, --brief for one-liners.
"""

import argparse
import json
import os
import sqlite3
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Query a DocDNA project knowledge database.",
        usage="%(prog)s /path/to/project [--subcommand] [args...]",
    )
    p.add_argument("project", nargs="?", default="", help="Path to the project root (contains DocDNA/)")
    p.add_argument("--mcp", action="store_true", help="Run as MCP stdio server (uses DOCDNA_ROOT env var)")
    p.add_argument("--json", action="store_true", help="Output structured JSON")
    p.add_argument("--brief", action="store_true", help="Minimal one-line output")

    # Subcommands (mutually exclusive group for the primary action)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--overview", action="store_true", help="Project summary")
    g.add_argument("--function", metavar="NAME", help="Function details by name")
    g.add_argument("--class", metavar="NAME", dest="class_name", help="Class details by name")
    g.add_argument("--search", metavar="TEXT", help="FTS5 full-text search across all knowledge")
    g.add_argument("--callgraph", metavar="NAME", help="Call graph for a function/class")
    g.add_argument("--file", metavar="PATH", help="All symbols in a file")
    g.add_argument("--faq", metavar="TOPIC", help="Search project FAQs")
    g.add_argument("--where-used", metavar="NAME", help="All references to a function/class")
    g.add_argument("--dead-code", action="store_true", help="List functions with zero callers")
    g.add_argument("--impact", metavar="NAME", help="Reverse transitive callers (all dependents)")
    g.add_argument("--imports", metavar="NAME", help="Import resolution for a symbol")
    g.add_argument("--category", metavar="NAME", help="All functions in a category")
    g.add_argument("--tags", metavar="TAG", help="All functions with a capability tag (talks-to-ai, edits-text, interactive, file-io)")
    g.add_argument("--dataflow", metavar="NAME", help="Show what a function does to the shared text/context, or 'all' for the full map")

    p.add_argument("--depth", metavar="N", type=int, default=1, help="Call graph depth (default 1)")
    p.add_argument("--limit", metavar="N", type=int, default=20, help="Max results (default 20)")

    return p


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def db_connect(project: str) -> sqlite3.Connection:
    """Locate DocDNA/docdna.db under *project* and return a connection."""
    candidates = [
        Path(project) / "DocDNA" / "docdna.db",
        Path(project) / "docdna.db",
    ]
    for path in candidates:
        if path.exists():
            return sqlite3.connect(str(path))
    sys.exit(f"Error: no docdna.db found under {project}/DocDNA/")


def fmt_md(heading: str, body: str, level: int = 2) -> str:
    return f"\n{'#' * level} {heading}\n\n{body.strip()}\n"


def fmt_json(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def json_or_md(data: dict, markdown: str, args) -> str:
    if args.json:
        return fmt_json(data)
    return markdown


def check_staleness(cur, project_path: Path) -> str:
    """Compare stored file hashes to current on-disk hashes.

    Returns a warning string if any indexed file has changed since the last
    DocDNA run, or an empty string if everything is current.
    """
    import hashlib
    try:
        cur.execute("SELECT path, content_hash FROM files WHERE content_hash IS NOT NULL")
        rows = cur.fetchall()
    except Exception:
        return ""
    if not rows:
        return ""
    stale = []
    for rel_path, stored_hash in rows:
        full_path = project_path / rel_path
        try:
            current_hash = hashlib.md5(full_path.read_bytes()).hexdigest()
            if current_hash != stored_hash:
                stale.append(rel_path)
        except FileNotFoundError:
            stale.append(rel_path)
        except Exception:
            pass
    if stale:
        files_list = ", ".join(stale[:5]) + (" ..." if len(stale) > 5 else "")
        return (
            f"\n⚠  STALE INDEX: {len(stale)} file(s) changed since last DocDNA run "
            f"({files_list}). Results may be incomplete. Ask the user to re-run DocDNA.\n"
        )
    return ""


# ---------------------------------------------------------------------------
# Subcommand: --overview
# ---------------------------------------------------------------------------

def cmd_overview(cur, args):
    stats = {}
    for table, label in [("functions", "functions"), ("classes", "classes"),
                          ("files", "files"), ("faqs", "FAQs"),
                          ("architecture_insights", "insights")]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        stats[label] = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM call_graph")
    stats["call_graph_edges"] = cur.fetchone()[0]

    # Architecture categories
    cur.execute("SELECT DISTINCT category FROM architecture_insights")
    cats = [r[0] for r in cur.fetchall()]

    # File list
    cur.execute("SELECT path, purpose FROM files ORDER BY path")
    files = [{"path": r[0], "purpose": r[1]} for r in cur.fetchall()]

    # Module overview
    cur.execute("SELECT DISTINCT module FROM imports ORDER BY module")
    modules = [r[0] for r in cur.fetchall()]

    data = {
        "type": "overview",
        "stats": stats,
        "architecture_categories": cats,
        "files": len(files),
        "modules": len(modules),
    }

    md = fmt_md("Project Overview", f"""
**Stats:** {stats['functions']} functions · {stats['classes']} classes · {stats['files']} files · {stats['call_graph_edges']} call edges · {stats['FAQs']} FAQs · {stats['insights']} insights

**Architecture categories:** {', '.join(cats) if cats else '*none*'}

**Files ({len(files)}):**
""" + "\n".join(f"  - `{f['path']}` — {f['purpose'] or '(no description)'}" for f in files[:20]) +
                    ("\n  - ..." if len(files) > 20 else "") +
                    f"\n\n**Modules ({len(modules)}):**\n  " + ', '.join(modules[:30]) +
                    ("\n  - ..." if len(modules) > 30 else ""))

    return json_or_md(data, md, args)


# ---------------------------------------------------------------------------
# Subcommand: --function
# ---------------------------------------------------------------------------

def cmd_function(cur, name, args):
    cur.execute("""
        SELECT name, file, line, args, docstring, decorators,
               is_method, return_type, inferred_purpose, confidence, source
        FROM functions WHERE name = ?
        ORDER BY file, line
    """, (name,))
    rows = cur.fetchall()

    if not rows:
        # Try partial match
        cur.execute("""
            SELECT name, file, line FROM functions
            WHERE name LIKE ? ORDER BY name LIMIT ?
        """, (f"%{name}%", args.limit))
        close = [{"name": r[0], "file": r[1], "line": r[2]} for r in cur.fetchall()]
        if close:
            msg = f"Function '{name}' not found. Did you mean?\n"
            for c in close:
                msg += f"  - {c['name']}  ({c['file']}:{c['line']})\n"
            return json_or_md({"type": "error", "message": f"Not found: {name}", "suggestions": close}, msg, args)
        return json_or_md({"type": "error", "message": f"Function not found: {name}"}, f"Function not found: {name}", args)

    results = []
    for r in rows:
        fn = {
            "name": r[0], "file": r[1], "line": r[2],
            "args": json.loads(r[3]) if r[3] else [],
            "docstring": r[4],
            "decorators": json.loads(r[5]) if r[5] else [],
            "is_method": bool(r[6]),
            "return_type": r[7],
            "inferred_purpose": r[8],
            "confidence": r[9],
            "source": r[10],
        }

        # Code snippet: first 10 lines of body
        cur.execute("SELECT content FROM code_content WHERE file = ? AND line_number >= ? ORDER BY line_number LIMIT 10",
                    (r[1], r[2]))
        snippet_lines = [s[0] for s in cur.fetchall()]
        fn["code_snippet"] = "\n".join(snippet_lines)

        results.append(fn)

    data = {"type": "function", "results": results[:args.limit]}

    md_lines = []
    for fn in results[:args.limit]:
        header = f"**`{fn['name']}`** → `{fn['file']}:{fn['line']}`"
        if fn["is_method"]:
            header += "  *(method)*"
        if fn["decorators"]:
            header += f"  `@{', @'.join(fn['decorators'])}`"
        md_lines.append(header)

        # Signature
        args_str = ", ".join(fn["args"])
        ret = f" -> {fn['return_type']}" if fn["return_type"] else ""
        md_lines.append(f"`def {fn['name']}({args_str}){ret}`")

        # Purpose
        purpose = fn["inferred_purpose"] or "(no purpose inferred)"
        conf = fn["confidence"] or ""
        if fn["docstring"]:
            md_lines.append(f"\n> {fn['docstring'].strip()}")
        md_lines.append(f"\n**Purpose:** {purpose}  *(confidence: {conf}, source: {fn['source']})*")

        # Code snippet
        if fn["code_snippet"]:
            md_lines.append(f"\n```python\n{fn['code_snippet']}\n```")
        md_lines.append("")

    return json_or_md(data, "\n".join(md_lines), args)


# ---------------------------------------------------------------------------
# Subcommand: --class
# ---------------------------------------------------------------------------

def cmd_class(cur, name, args):
    cur.execute("""
        SELECT c.name, c.file, c.line, c.bases, c.docstring, c.methods, c.inferred_purpose,
               m.name, m.line, m.args, m.docstring, m.return_type
        FROM classes c
        LEFT JOIN functions m ON m.file = c.file AND m.name IN (
            SELECT value FROM json_each(c.methods)
        )
        WHERE c.name = ?
        ORDER BY m.line
    """, (name,))

    rows = cur.fetchall()
    if not rows:
        return json_or_md({"type": "error", "message": f"Class not found: {name}"}, f"Class not found: {name}", args)

    cls = {
        "name": rows[0][0], "file": rows[0][1], "line": rows[0][2],
        "bases": json.loads(rows[0][3]) if rows[0][3] else [],
        "docstring": rows[0][4],
        "methods": [],
    }
    seen_methods = set()
    for r in rows:
        if r[7] and r[7] not in seen_methods:
            seen_methods.add(r[7])
            cls["methods"].append({
                "name": r[7], "line": r[8], "args": json.loads(r[9]) if r[9] else [],
                "docstring": r[10], "return_type": r[11],
            })

    data = {"type": "class", "result": cls}

    bases_str = ", ".join(cls["bases"]) if cls["bases"] else "(none)"
    md = fmt_md(f"Class `{cls['name']}`", f"""
**File:** `{cls['file']}:{cls['line']}`
**Inherits:** {bases_str}
**Docstring:** {cls['docstring'] or '*none*'}

**Methods ({len(cls['methods'])}):**
""" + "\n".join(
        f"  - `{m['name']}` at line {m['line']} → `({', '.join(m['args'])})`"
        + (f" -> {m['return_type']}" if m["return_type"] else "")
        + (f" — {m['docstring'].split(chr(10))[0]}" if m["docstring"] else "")
        for m in cls["methods"]
    ) if cls["methods"] else "  *(no methods found)*")

    return json_or_md(data, md, args)


# ---------------------------------------------------------------------------
# Subcommand: --search (FTS5)
# ---------------------------------------------------------------------------

def cmd_search(cur, text, args):
    # Sanitize FTS5 query: escape special chars, wrap terms in quotes if they contain spaces
    fts_query = text.replace('"', '""')
    # Wrap in double quotes for phrase search
    fts_query = f'"{fts_query}"'

    results = []

    # Search functions_fts
    try:
        cur.execute("""
            SELECT 'function' as kind, f.name, f.file, f.line,
                   snippet(functions_fts, 2, '<mark>', '</mark>', '...', 40) as snippet,
                   rank
            FROM functions_fts
            JOIN functions f ON f.id = functions_fts.rowid
            WHERE functions_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, args.limit))
        for r in cur.fetchall():
            results.append({
                "kind": "function", "name": r[1], "file": r[2], "line": r[3],
                "snippet": r[4], "rank": r[5],
            })
    except sqlite3.OperationalError:
        pass  # no matches

    # Search code_fts
    try:
        cur.execute("""
            SELECT 'code' as kind, cc.file, cc.line_number,
                   snippet(code_fts, 1, '<mark>', '</mark>', '...', 40) as snippet,
                   rank
            FROM code_fts
            JOIN code_content cc ON cc.id = code_fts.rowid
            WHERE code_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, args.limit - len(results)))
        for r in cur.fetchall():
            results.append({
                "kind": "code", "name": "", "file": r[1], "line": r[2],
                "snippet": r[3], "rank": r[4],
            })
    except sqlite3.OperationalError:
        pass

    # Search faqs_fts
    try:
        cur.execute("""
            SELECT 'faq' as kind, f.question, '', 0,
                   snippet(faqs_fts, 1, '<mark>', '</mark>', '...', 60) as snippet,
                   rank
            FROM faqs_fts
            JOIN faqs f ON f.id = faqs_fts.rowid
            WHERE faqs_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, max(5, args.limit - len(results))))
        for r in cur.fetchall():
            results.append({
                "kind": "faq", "name": r[1], "file": "", "line": 0,
                "snippet": r[4], "rank": r[5],
            })
    except sqlite3.OperationalError:
        pass

    # Sort by rank
    results.sort(key=lambda x: x.get("rank", 999))

    if not results:
        # Fallback: LIKE search on function names
        cur.execute("""
            SELECT name, file, line, inferred_purpose
            FROM functions WHERE name LIKE ? OR inferred_purpose LIKE ?
            LIMIT ?
        """, (f"%{text}%", f"%{text}%", args.limit))
        like_results = [{"kind": "like", "name": r[0], "file": r[1], "line": r[2],
                         "snippet": r[3] or "", "rank": 999} for r in cur.fetchall()]
        if like_results:
            results = like_results
        else:
            return json_or_md({"type": "error", "message": f"No matches for: {text}"},
                              f"No matches found for: {text}", args)

    data = {"type": "search", "query": text, "results": results}

    md = fmt_md(f"Search results for \"{text}\"", "")
    kind_labels = {"function": "Functions", "code": "Code", "faq": "FAQs", "like": "Name/Inferred match"}
    for kind in ["function", "code", "faq", "like"]:
        items = [r for r in results if r["kind"] == kind]
        if not items:
            continue
        md += f"\n**{kind_labels[kind]}:**\n"
        for r in items[:10]:
            loc = f"`{r['file']}:{r['line']}`" if r.get("file") else ""
            name_str = f"`{r['name']}` " if r.get("name") else ""
            md += f"  - {name_str}{loc} — {r['snippet']}\n"

    return json_or_md(data, md, args)


# ---------------------------------------------------------------------------
# Subcommand: --callgraph
# ---------------------------------------------------------------------------

def cmd_callgraph(cur, name, depth, args):
    # Find callers (who calls NAME)
    callers = []
    cur.execute("""
        SELECT DISTINCT caller, caller_file FROM call_graph WHERE callee = ?
        ORDER BY caller_file, caller
    """, (name,))
    callers = [{"name": r[0], "file": r[1]} for r in cur.fetchall()]

    # Find callees (who NAME calls)
    callees = []
    cur.execute("""
        SELECT DISTINCT callee, callee_file FROM call_graph WHERE caller = ?
        ORDER BY callee_file, callee
    """, (name,))
    callees = [{"name": r[0], "file": r[1]} for r in cur.fetchall()]

    # Depth > 1: transitive callers (reverse)
    transitive_callers = []
    if depth > 1:
        _gather_transitive(cur, name, "caller", set(), transitive_callers, depth - 1, 1)

    data = {
        "type": "callgraph", "name": name,
        "callers": callers, "callees": callees,
        "transitive_callers": transitive_callers,
    }

    md = []
    md.append(f"**Call graph for `{name}`**\n")

    if callers:
        md.append(f"**Called by ({len(callers)}):**")
        for c in callers:
            md.append(f"  - `{c['name']}`  → `{c['file']}`")
    else:
        md.append("**Called by:** *(no callers found — may be dead code or entry point)*")

    if transitive_callers:
        indent_map = {1: "  ", 2: "    ", 3: "      "}
        for t in transitive_callers:
            ind = indent_map.get(t["depth"], "  ")
            md.append(f"{ind}`{t['name']}`  → `{t['file']}`")

    md.append("")
    if callees:
        md.append(f"**Calls ({len(callees)}):**")
        for c in callees:
            md.append(f"  - `{c['name']}`  → `{c['file'] or '(builtin/external)'}`")
    else:
        md.append("**Calls:** *(no callees recorded)*")

    data["has_callers"] = len(callers) > 0
    data["has_callees"] = len(callees) > 0

    return json_or_md(data, "\n".join(md), args)


def _gather_transitive(cur, name, direction, seen, results, remaining_depth, current_depth):
    """Recursively gather transitive callers/callees."""
    if remaining_depth <= 0:
        return
    col = "caller" if direction == "caller" else "callee"
    target = "callee" if direction == "caller" else "caller"
    cur.execute(f"SELECT DISTINCT {col}, {col}_file FROM call_graph WHERE {target} = ?", (name,))
    for r in cur.fetchall():
        key = (r[0], r[1] or "")
        if key in seen:
            continue
        seen.add(key)
        results.append({"name": r[0], "file": r[1] or "", "depth": current_depth})
        _gather_transitive(cur, r[0], direction, seen, results, remaining_depth - 1, current_depth + 1)


# ---------------------------------------------------------------------------
# Subcommand: --file
# ---------------------------------------------------------------------------

def cmd_file(cur, file_path, args):
    # Normalize path (strip leading ./ or ../)
    file_path = file_path.replace("\\", "/").lstrip("./")

    # Try to find the file
    cur.execute("SELECT path, line_count, docstring, module_name, purpose FROM files WHERE path LIKE ? OR path = ?",
                (f"%{file_path}%", file_path))
    file_row = cur.fetchone()

    if not file_row:
        # List matching files
        cur.execute("SELECT path FROM files WHERE path LIKE ? LIMIT ?", (f"%{file_path}%", args.limit))
        matches = [r[0] for r in cur.fetchall()]
        if matches:
            return json_or_md({"type": "error", "message": f"File not found: {file_path}", "suggestions": matches},
                              f"File not found: '{file_path}'. Did you mean?\n" + "\n".join(f"  - {m}" for m in matches), args)
        return json_or_md({"type": "error", "message": f"File not found: {file_path}"},
                          f"File not found: {file_path}", args)

    path = file_row[0]
    line_count = file_row[1]

    # Functions in this file
    cur.execute("""
        SELECT name, line, args, docstring, return_type, is_method, decorators, inferred_purpose
        FROM functions WHERE file = ? ORDER BY line
    """, (path,))
    funcs = [{
        "name": r[0], "line": r[1], "args": json.loads(r[2]) if r[2] else [],
        "docstring": r[3], "return_type": r[4], "is_method": bool(r[5]),
        "decorators": json.loads(r[6]) if r[6] else [],
        "purpose": r[7],
    } for r in cur.fetchall()]

    # Classes in this file
    cur.execute("""
        SELECT name, line, bases, docstring, methods FROM classes WHERE file = ? ORDER BY line
    """, (path,))
    classes = [{
        "name": r[0], "line": r[1], "bases": json.loads(r[2]) if r[2] else [],
        "docstring": r[3], "methods": json.loads(r[4]) if r[4] else [],
    } for r in cur.fetchall()]

    # Imports in this file
    cur.execute("""
        SELECT module, alias, from_module FROM imports WHERE file = ? ORDER BY line_number
    """, (path,))
    imports = [{"module": r[0], "alias": r[1], "from": r[2]} for r in cur.fetchall()]

    data = {
        "type": "file", "file": path, "lines": line_count,
        "docstring": file_row[2], "module": file_row[3], "purpose": file_row[4],
        "functions": len(funcs), "classes": len(classes), "imports": len(imports),
    }

    md = [fmt_md(f"File: `{path}`", f"""
**Lines:** {line_count or '?'}
**Module:** {file_row[3] or '*unknown*'}
**Purpose:** {file_row[4] or '(not documented)'}
**Functions:** {len(funcs)} · **Classes:** {len(classes)} · **Imports:** {len(imports)}
""")]

    if classes:
        md.append("**Classes:**")
        for c in classes:
            bases = f"({', '.join(c['bases'])})" if c["bases"] else ""
            md.append(f"  - `{c['name']}` {bases} at line {c['line']} — {c['docstring'] or c.get('purpose', '')[:80]}")
        md.append("")

    if funcs:
        md.append("**Functions:**")
        for f in funcs:
            kind = " *(method)*" if f["is_method"] else ""
            dec = f" @{', @'.join(f['decorators'])}" if f["decorators"] else ""
            ret = f" → {f['return_type']}" if f["return_type"] else ""
            args_str = ", ".join(f["args"])
            desc = f["docstring"] or f["purpose"] or ""
            md.append(f"  - `{f['name']}({args_str}){ret}` at line {f['line']}{kind}{dec} — {desc[:120]}")
        md.append("")

    if imports:
        md.append(f"**Imports ({len(imports)}):**")
        for im in imports:
            src = f" from {im['from']}" if im["from"] else ""
            alias = f" as {im['alias']}" if im["alias"] else ""
            md.append(f"  - `{im['module']}`{alias}{src}")

    return json_or_md(data, "\n".join(md), args)


# ---------------------------------------------------------------------------
# Subcommand: --faq
# ---------------------------------------------------------------------------

def cmd_faq(cur, topic, args):
    # Try FTS5 first
    try:
        fts_query = f'"{topic}"'
        cur.execute("""
            SELECT question, answer, refs FROM faqs_fts
            JOIN faqs f ON f.id = faqs_fts.rowid
            WHERE faqs_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, args.limit))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []

    if not rows:
        # LIKE fallback
        cur.execute("""
            SELECT question, answer, refs FROM faqs
            WHERE question LIKE ? OR answer LIKE ?
            LIMIT ?
        """, (f"%{topic}%", f"%{topic}%", args.limit))
        rows = cur.fetchall()

    if not rows:
        # Show all FAQs as browse
        cur.execute("SELECT question, answer, refs FROM faqs ORDER BY question LIMIT ?", (args.limit,))
        rows = cur.fetchall()
        if not rows:
            return json_or_md({"type": "error", "message": "No FAQs in database"},
                              "No FAQs in database", args)
        prefix = "(showing all FAQs — use --faq <topic> to narrow)\n\n"
    else:
        prefix = ""

    results = [{"question": r[0], "answer": r[1],
                "refs": json.loads(r[2]) if r[2] else []} for r in rows]

    data = {"type": "faq", "results": results}

    md = prefix
    for r in results:
        refs_str = "  \n    Refs: " + ", ".join(r["refs"]) if r["refs"] else ""
        md += f"**Q:** {r['question']}\n**A:** {r['answer']}{refs_str}\n\n---\n\n"

    if args.brief:
        md = prefix + "\n".join(f"  - {r['question'][:100]}" for r in results)

    return json_or_md(data, md, args)


# ---------------------------------------------------------------------------
# Subcommand: --where-used
# ---------------------------------------------------------------------------

def cmd_where_used(cur, name, args):
    parts = []

    # Call graph references (as callee)
    cur.execute("""
        SELECT caller, caller_file, line_number FROM call_graph WHERE callee = ?
        LIMIT ?
    """, (name, args.limit))
    for r in cur.fetchall():
        parts.append({"kind": "caller", "name": r[0], "file": r[1], "line": r[2] or 0})

    # Imports that reference the name
    cur.execute("""
        SELECT file, line_number, module FROM imports
        WHERE module = ? OR from_module = ?
        LIMIT ?
    """, (name, name, args.limit - len(parts)))
    for r in cur.fetchall():
        parts.append({"kind": "import", "name": r[2], "file": r[0], "line": r[1] or 0})

    if not parts:
        return json_or_md({"type": "error", "message": f"No references to: {name}"},
                          f"No references found for: {name}", args)

    data = {"type": "where_used", "name": name, "references": parts}
    md = [f"**References to `{name}` ({len(parts)}):**"]
    for p in parts:
        loc = f"`{p['file']}:{p['line']}`" if p["file"] else ""
        md.append(f"  - [{p['kind']}] {loc}  `{p['name']}`")

    return json_or_md(data, "\n".join(md), args)


# ---------------------------------------------------------------------------
# Subcommand: --dead-code
# ---------------------------------------------------------------------------

def cmd_dead_code(cur, args):
    # Functions with zero callers in the call_graph
    cur.execute("""
        SELECT f.name, f.file, f.line, f.inferred_purpose
        FROM functions f
        WHERE f.name NOT IN (SELECT DISTINCT callee FROM call_graph)
          AND f.name NOT IN (SELECT DISTINCT caller FROM call_graph)
        ORDER BY f.file, f.line
        LIMIT ?
    """, (args.limit,))
    rows = cur.fetchall()

    if not rows:
        return json_or_md({"type": "dead_code", "message": "No dead code found"},
                          "No dead code found — every function has at least one call graph entry", args)

    results = [{"name": r[0], "file": r[1], "line": r[2], "purpose": r[3]} for r in rows]
    data = {"type": "dead_code", "functions": results}

    md = fmt_md(f"Potentially Dead Code ({len(results)} functions)", "")
    for r in results:
        purpose = r["purpose"] or "(unknown)"
        md += f"  - `{r['name']}` → `{r['file']}:{r['line']}` — {purpose[:80]}\n"

    return json_or_md(data, md, args)


# ---------------------------------------------------------------------------
# Subcommand: --impact
# ---------------------------------------------------------------------------

def cmd_impact(cur, name, args):
    # Gather all transitive callers (what breaks if this function changes)
    transitive = []
    _gather_transitive(cur, name, "caller", set(), transitive, args.depth, 1)

    # Also include direct callers
    cur.execute("""
        SELECT DISTINCT caller, caller_file FROM call_graph WHERE callee = ?
        ORDER BY caller_file, caller
    """, (name,))
    direct_callers = [{"name": r[0], "file": r[1]} for r in cur.fetchall()]

    if not direct_callers and not transitive:
        return json_or_md({"type": "error", "message": f"No dependents found for: {name}"},
                          f"No dependents found for: {name}", args)

    all_callers = [{"name": c["name"], "file": c["file"], "depth": 0} for c in direct_callers] + transitive

    data = {"type": "impact", "name": name, "dependents": all_callers}
    md = [f"**Impact analysis for `{name}` — {len(all_callers)} dependents**\n"]
    md.append("If this function changes, the following may break:\n")
    for c in all_callers:
        depth_mark = "└─" if c.get("depth", 0) else "  "
        prefix = "  " * (c.get("depth", 0)) + depth_mark
        md.append(f"  {prefix} `{c['name']}` → `{c['file']}`")

    return json_or_md(data, "\n".join(md), args)


# ---------------------------------------------------------------------------
# Subcommand: --imports
# ---------------------------------------------------------------------------

def cmd_imports(cur, name, args):
    cur.execute("""
        SELECT file, line_number, module, alias, from_module
        FROM imports WHERE module = ? OR from_module = ? OR alias = ?
        LIMIT ?
    """, (name, name, name, args.limit))
    rows = cur.fetchall()

    if not rows:
        return json_or_md({"type": "error", "message": f"No imports found for: {name}"},
                          f"No imports found for: {name}", args)

    results = [{"file": r[0], "line": r[1], "module": r[2], "alias": r[3], "from": r[4]} for r in rows]
    data = {"type": "imports", "name": name, "imports": results}

    md = [f"**Import references for `{name}`:**\n"]
    for r in results:
        src = f" from {r['from']}" if r["from"] else ""
        alias = f" as {r['alias']}" if r["alias"] else ""
        md.append(f"  - `{r['module']}`{alias}{src}  → `{r['file']}:{r['line'] or '?'}`")

    return json_or_md(data, "\n".join(md), args)


# ---------------------------------------------------------------------------
# Subcommand: --category
# ---------------------------------------------------------------------------

def cmd_tags(cur, tag, args):
    """List all functions carrying a given capability tag."""
    KNOWN_TAGS = ["talks-to-ai", "edits-text", "interactive", "file-io"]

    # If tag is "list" or empty, show available tags and counts
    if not tag or tag == "list":
        cur.execute("SELECT tag, COUNT(*) FROM function_tags GROUP BY tag ORDER BY tag")
        rows = cur.fetchall()
        if not rows:
            return json_or_md({"type": "tags", "available": []},
                              f"No capability tags found. Re-run DocDNA to generate them.\nKnown tags: {', '.join(KNOWN_TAGS)}", args)
        summary = {r[0]: r[1] for r in rows}
        md = ["**Capability tag counts:**"]
        for t, n in summary.items():
            md.append(f"  - `{t}`: {n} functions")
        return json_or_md({"type": "tags", "available": summary}, "\n".join(md), args)

    cur.execute("""
        SELECT t.name, t.file, f.line, f.inferred_purpose
        FROM function_tags t
        LEFT JOIN functions f ON f.name = t.name AND f.file = t.file
        WHERE t.tag = ?
        ORDER BY t.file, f.line
        LIMIT ?
    """, (tag, args.limit))
    rows = cur.fetchall()

    if not rows:
        return json_or_md({"type": "error", "message": f"No functions found with tag: {tag}"},
                          f"No functions tagged `{tag}`. Known tags: {', '.join(KNOWN_TAGS)}", args)

    funcs = [{"name": r[0], "file": r[1], "line": r[2], "purpose": r[3]} for r in rows]
    data = {"type": "tags", "tag": tag, "functions": funcs}
    md = [fmt_md(f"Functions tagged `{tag}`", f"{len(funcs)} found\n")]
    for f in funcs:
        md.append(f"  - `{f['name']}` → `{f['file']}:{f['line'] or '?'}` — {f['purpose'] or ''}")
    return json_or_md(data, "\n".join(md), args)


def cmd_dataflow(cur, name, args):
    """Show what a function does to shared text/context, or the full map with 'all'."""
    if name == "all":
        cur.execute("""
            SELECT d.name, d.file, d.action, d.note
            FROM data_flow d
            ORDER BY d.action, d.file, d.name
            LIMIT ?
        """, (args.limit,))
        rows = cur.fetchall()
        if not rows:
            return json_or_md({"type": "dataflow", "entries": []},
                              "No data-flow entries found. Re-run DocDNA to generate them.", args)
        entries = [{"name": r[0], "file": r[1], "action": r[2], "note": r[3]} for r in rows]
        data = {"type": "dataflow", "entries": entries}
        md = [fmt_md("Data-flow map", f"{len(entries)} functions tracked\n")]
        for e in entries:
            md.append(f"  [{e['action']}] `{e['name']}` (`{e['file']}`) — {e['note'] or ''}")
        return json_or_md(data, "\n".join(md), args)

    cur.execute("""
        SELECT action, note FROM data_flow WHERE name = ?
    """, (name,))
    rows = cur.fetchall()
    if not rows:
        return json_or_md({"type": "error", "message": f"No data-flow entry for: {name}"},
                          f"No data-flow entry found for `{name}`. Either it doesn't touch text/context, or DocDNA needs a re-run.", args)
    data = {"type": "dataflow", "name": name, "entries": [{"action": r[0], "note": r[1]} for r in rows]}
    md = [fmt_md(f"Data-flow for `{name}`", "")]
    for r in rows:
        md.append(f"  Action: **{r[0]}** — {r[1] or ''}")
    return json_or_md(data, "\n".join(md), args)


def cmd_category(cur, cat, args):
    # categories from architecture_insights
    cur.execute("SELECT DISTINCT category FROM architecture_insights ORDER BY category")
    all_cats = [r[0] for r in cur.fetchall()]

    if cat not in all_cats:
        return json_or_md({"type": "error", "message": f"Category not found: {cat}", "available": all_cats},
                          f"Category '{cat}' not found. Available: {', '.join(all_cats)}", args)

    cur.execute("SELECT description, details, file FROM architecture_insights WHERE category = ?", (cat,))
    insights = [{"description": r[0], "details": json.loads(r[1]) if r[1] else {}, "file": r[2]} for r in cur.fetchall()]

    # Also find functions with matching categories/tags
    cur.execute("""
        SELECT name, file, line, inferred_purpose
        FROM functions WHERE inferred_purpose LIKE ? OR file LIKE ?
        LIMIT ?
    """, (f"%{cat}%", f"%{cat}%", args.limit))
    funcs = [{"name": r[0], "file": r[1], "line": r[2], "purpose": r[3]} for r in cur.fetchall()]

    data = {"type": "category", "category": cat, "insights": insights, "related_functions": funcs}

    md = [fmt_md(f"Category: {cat}", "")]
    for ins in insights:
        md.append(f"  - {ins['description']}")
        if "functions" in ins["details"]:
            md.append(f"    Functions: {', '.join(ins['details']['functions'])}")
    if funcs:
        md.append(f"\n**Related functions ({len(funcs)}):**")
        for f in funcs:
            md.append(f"  - `{f['name']}` → `{f['file']}:{f['line']}` — {f['purpose'] or ''}")

    return json_or_md(data, "\n".join(md), args)


# ---------------------------------------------------------------------------
# MCP server mode
# ---------------------------------------------------------------------------

def _mcp_tools_list() -> list:
    """Return the canonical MCP tools list (used by both initialize and tools/list)."""
    return [
        {"name": "docdna_overview", "description": "Project overview — stats, files, modules, architecture summary",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "docdna_function", "description": "Full details for a named function (signature, docstring, file:line)",
         "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        {"name": "docdna_class", "description": "Class details including methods and docstring",
         "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        {"name": "docdna_search", "description": "Full-text search across functions, code, FAQs, and instruction files",
         "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "docdna_callgraph", "description": "Callers and callees for a function; add depth for transitive",
         "inputSchema": {"type": "object", "properties": {
             "name": {"type": "string"}, "depth": {"type": "integer", "default": 1}}, "required": ["name"]}},
        {"name": "docdna_file", "description": "All functions, classes, and imports in a file",
         "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "docdna_faq", "description": "Search project FAQs and AI instruction file prose",
         "inputSchema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}},
        {"name": "docdna_where_used", "description": "Every file+line referencing a function or class",
         "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        {"name": "docdna_dead_code", "description": "Functions with zero callers",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "docdna_impact", "description": "All transitive callers — what breaks if this function changes",
         "inputSchema": {"type": "object", "properties": {
             "name": {"type": "string"}, "depth": {"type": "integer", "default": 3}}, "required": ["name"]}},
        {"name": "docdna_imports", "description": "Import resolution for a module or symbol",
         "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        {"name": "docdna_tags", "description": "Functions by capability tag: talks-to-ai, edits-text, interactive, file-io",
         "inputSchema": {"type": "object", "properties": {"tag": {"type": "string", "description": "Tag name or 'list'"}}}},
        {"name": "docdna_dataflow", "description": "What a function does to shared text/context, or 'all' for the full map",
         "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    ]


def _mcp_dispatch(cur, tool_name: str, args_in: dict) -> str:
    """Dispatch a tools/call request to the right command function."""
    ns_json = argparse.Namespace(json=True, limit=50, depth=args_in.get("depth", 1), brief=False)
    if tool_name == "docdna_overview":
        return cmd_overview(cur, ns_json)
    elif tool_name == "docdna_function":
        return cmd_function(cur, args_in.get("name", ""), ns_json)
    elif tool_name == "docdna_class":
        return cmd_class(cur, args_in.get("name", ""), ns_json)
    elif tool_name == "docdna_search":
        return cmd_search(cur, args_in.get("query", ""), argparse.Namespace(json=True, limit=20, brief=False))
    elif tool_name == "docdna_callgraph":
        return cmd_callgraph(cur, args_in.get("name", ""), args_in.get("depth", 1), ns_json)
    elif tool_name == "docdna_file":
        return cmd_file(cur, args_in.get("path", ""), ns_json)
    elif tool_name == "docdna_faq":
        return cmd_faq(cur, args_in.get("topic", ""), argparse.Namespace(json=True, limit=20, brief=False))
    elif tool_name == "docdna_where_used":
        return cmd_where_used(cur, args_in.get("name", ""), ns_json)
    elif tool_name == "docdna_dead_code":
        return cmd_dead_code(cur, ns_json)
    elif tool_name == "docdna_impact":
        return cmd_impact(cur, args_in.get("name", ""), argparse.Namespace(json=True, limit=50, depth=args_in.get("depth", 3), brief=False))
    elif tool_name == "docdna_imports":
        return cmd_imports(cur, args_in.get("name", ""), ns_json)
    elif tool_name == "docdna_tags":
        return cmd_tags(cur, args_in.get("tag", "list"), ns_json)
    elif tool_name == "docdna_dataflow":
        return cmd_dataflow(cur, args_in.get("name", "all"), ns_json)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def mcp_serve(project: str):
    """MCP 2024-11-05 server — reads JSON-RPC from stdin, writes to stdout.

    Handles initialize, tools/list, tools/call, and notifications/initialized.
    Register in Claude Code settings.json under mcpServers with:
        command: python3 /path/to/docdna_query.py --mcp
        env: {DOCDNA_ROOT: /path/to/project}
    """
    conn = db_connect(project)
    cur = conn.cursor()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method", "")

        try:
            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "docdna-query", "version": "2.0.0"},
                        "capabilities": {"tools": {}},
                    }
                }
                print(json.dumps(resp), flush=True)

            elif method == "notifications/initialized":
                pass  # No response to notifications

            elif method == "tools/list":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _mcp_tools_list()}}
                print(json.dumps(resp), flush=True)

            elif method == "tools/call":
                tool_name = req.get("params", {}).get("name", "")
                args_in = req.get("params", {}).get("arguments", {}) or {}
                result_text = _mcp_dispatch(cur, tool_name, args_in)
                resp = {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": result_text}]}
                }
                print(json.dumps(resp), flush=True)

            elif req_id is not None:
                # Unknown method with an id — send method not found
                resp = {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }
                print(json.dumps(resp), flush=True)

        except Exception as e:
            if req_id is not None:
                resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}}
                print(json.dumps(resp), flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = build_parser()
    args = p.parse_args()

    project = args.project

    # MCP mode
    if args.mcp:
        # Accept project path from positional arg, DOCDNA_ROOT env var, or fail
        project = args.project or os.environ.get("DOCDNA_ROOT", "")
        if not project:
            sys.exit("Error: project path required for --mcp mode (pass as argument or set DOCDNA_ROOT)")
        mcp_serve(project)
        return

    conn = db_connect(project)
    cur = conn.cursor()

    # Staleness check — warn if source files changed since last index
    project_path = Path(project).resolve()
    stale_warning = check_staleness(cur, project_path)

    # Dispatch subcommand
    if args.overview:
        output = cmd_overview(cur, args)
    elif args.function:
        output = cmd_function(cur, args.function, args)
    elif args.class_name:
        output = cmd_class(cur, args.class_name, args)
    elif args.search:
        output = cmd_search(cur, args.search, args)
    elif args.callgraph:
        output = cmd_callgraph(cur, args.callgraph, args.depth, args)
    elif args.file:
        output = cmd_file(cur, args.file, args)
    elif args.faq:
        output = cmd_faq(cur, args.faq, args)
    elif args.where_used:
        output = cmd_where_used(cur, args.where_used, args)
    elif args.dead_code:
        output = cmd_dead_code(cur, args)
    elif args.impact:
        output = cmd_impact(cur, args.impact, args)
    elif args.imports:
        output = cmd_imports(cur, args.imports, args)
    elif args.category:
        output = cmd_category(cur, args.category, args)
    elif args.tags:
        output = cmd_tags(cur, args.tags, args)
    elif args.dataflow:
        output = cmd_dataflow(cur, args.dataflow, args)
    else:
        # No subcommand = default to overview
        output = cmd_overview(cur, args)

    conn.close()
    if stale_warning:
        print(stale_warning)
    print(output)


if __name__ == "__main__":
    main()

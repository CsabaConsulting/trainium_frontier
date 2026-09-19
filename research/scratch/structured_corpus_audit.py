"""Read-only structure coverage audit of the five permitted training shards.

This inspects complete documents for research triage, not model predictions.
Document/fence parsing must NEVER be used to produce earlier-token logits:
that would read future tokens. No document content is executed or saved.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import warnings

import pyarrow.parquet as pq


FENCE = re.compile(
    r"^ {0,3}(`{3,}|~{3,})([^\n]*)\n(.*?)^ {0,3}\1[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
PYTHON_LINE = re.compile(
    r"^\s*(?:from [A-Za-z_]\w*(?:\.\w+)* import |"
    r"import [A-Za-z_]\w*|(?:async )?def [A-Za-z_]\w*\s*\(|"
    r"class [A-Za-z_]\w*(?:\([^\n]*\))?\s*:)", re.MULTILINE,
)
JSON_KEY = re.compile(r'\{\s*"[^"\n]{1,128}"\s*:')
SCRIPTS = {
    "cjk": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]"),
    "cyrillic": re.compile(r"[\u0400-\u052f]"),
    "arabic": re.compile(r"[\u0600-\u06ff]"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parses(body, language):
    # Bounded parsing only; no eval, exec, compilation to bytecode or imports.
    if not body.strip() or len(body) > 16384:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            if language == "python":
                ast.parse(body, mode="exec")
            else:
                json.loads(body)
        return True
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return False


def inspect(text, counts, examples, source, row):
    counts["documents"] += 1
    counts["utf8_bytes"] += len(text.encode("utf-8"))
    counts["characters"] += len(text)
    counts["documents_with_non_ascii"] += not text.isascii()
    for name, pattern in SCRIPTS.items():
        counts["documents_with_" + name + "_characters"] += bool(pattern.search(text))
    counts["documents_with_python_line_heuristic"] += bool(PYTHON_LINE.search(text))
    counts["documents_with_json_key_heuristic"] += bool(JSON_KEY.search(text))
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        result = parses(stripped, "json")
        counts["whole_document_json_candidates"] += 1
        counts["whole_document_json_parsed"] += result is True
        counts["whole_document_json_skipped_length"] += result is None
    found = set()
    for match in FENCE.finditer(text):
        info, body = match.group(2).strip().lower(), match.group(3)
        tag = info.split()[0] if info else ""
        counts["fenced_blocks"] += 1
        language = ("python" if tag in ("py", "python", "python3", "py3") else
                    "json" if tag == "json" else None)
        if language is None:
            continue
        found.add(language)
        counts[language + "_fenced_blocks"] += 1
        counts[language + "_fenced_utf8_bytes"] += len(body.encode("utf-8"))
        result = parses(body, language)
        counts[language + "_fences_parsed"] += result is True
        counts[language + "_fences_skipped_length"] += result is None
        if result is True and len(examples[language]) < 8:
            examples[language].append(dict(shard=source, document_index=row,
                span=list(match.span()), utf8_bytes=len(body.encode("utf-8")),
                body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest()))
    for language in found:
        counts["documents_with_" + language + "_fence"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--documents-per-shard", type=int, default=4000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.documents_per_shard <= 10000
    assert not args.output.exists(), "Do not overwrite an earlier audit"
    started = time.perf_counter()
    total, shards = Counter(), []
    examples = {"python": [], "json": []}
    for index in range(5):
        path = args.directory / f"shard_{index:05d}.parquet"
        before = path.stat()
        authority = digest(path)
        data = pq.ParquetFile(path)
        counts = Counter()
        for batch in data.iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column(0).to_pylist():
                if counts["documents"] >= args.documents_per_shard:
                    break
                assert isinstance(text, str)
                inspect(text, counts, examples, path.name, counts["documents"])
            if counts["documents"] >= args.documents_per_shard:
                break
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        total.update(counts)
        shards.append(dict(path=str(path), sha256=authority, bytes=before.st_size,
            available_documents=data.metadata.num_rows, counts=dict(counts)))
    result = dict(at=datetime.now(timezone.utc).isoformat(), read_only=True,
        source_sha256=digest(Path(__file__)), seconds=time.perf_counter()-started,
        documents_per_shard=args.documents_per_shard, counts=dict(total),
        shards=shards, parsed_examples_without_content=examples,
        limitations=["Prefix sample from each permitted training shard; no validation used.",
            "Regexes miss unfenced and differently formatted code; heuristics have false positives.",
            "Script/non-ASCII presence is not language identification.",
            "AST acceptance is for complete fences, not validity of partial Python prefixes.",
            "No measured BPB gain or permission to build an uncharged corpus index."])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=str(args.output), seconds=result['seconds'], counts=result['counts'])))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read installed NLP-related distribution metadata without importing packages.

No model, dictionary, corpus, or device is opened. Resource hints are filenames
from wheel metadata, not proof of usable assets or permission to use them.
"""
import datetime
import importlib.metadata as metadata
import json
import re
import sys


NAMES = """
nltk spacy thinc textblob gensim stanza flair pattern pattern3 polyglot textacy
nlpaug snowballstemmer stemming pystemmer symspellpy pyspellchecker pyenchant
wordfreq wordninja wordsegment cmudict pronouncing sacremoses sacrebleu kenlm
langdetect langid fasttext fasttext-wheel regex tokenizers sentencepiece tiktoken
transformers marisa-trie datrie pyahocorasick rapidfuzz fuzzywuzzy
python-levenshtein levenshtein pyparsing lark interegular babel unidecode
text-unidecode xgrammar outlines outlines-core lm-format-enforcer
""".split()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def inventory():
    installed = {canonical(d.metadata["Name"]): d for d in metadata.distributions()
                 if d.metadata.get("Name")}
    parents = {}
    for name, dist in installed.items():
        for requirement in dist.requires or []:
            match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
            if match:
                parents.setdefault(canonical(match.group(1)), []).append(
                    {"distribution": name, "declared_requirement": requirement})
    hints = re.compile(r"english|wordlist|wordfreq|dictionary|lexicon|stopword|"
                       r"vocab|merges|corpus|tagger|stemmer", re.I)
    selected = {}
    for name in NAMES:
        dist = installed.get(canonical(name))
        result = {"installed": dist is not None,
                  "declared_by": parents.get(canonical(name), [])}
        if dist is not None:
            resources = [str(p) for p in dist.files or [] if hints.search(str(p))]
            result.update(version=dist.version, resource_filename_hint_count=len(resources),
                          resource_filename_hints=resources[:32])
        selected[name] = result
    return {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "installed_distribution_count": len(installed),
        "all_distribution_versions": {n: d.version for n, d in sorted(installed.items())},
        "selected": selected,
        "scope": "Active venv wheel metadata; selected names and filename hints, not an exhaustive resource audit",
        "dependency_edge_caveat": "Declared requirements include optional extras and unevaluated environment markers; they are not a resolved installation provenance graph",
        "target_packages_imported": False, "assets_opened": False,
        "device_used": False, "official_submission_environment_verified": False,
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))

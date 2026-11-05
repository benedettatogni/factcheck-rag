"""Loading fact-checks and test queries.

Two formats are supported:
1. Simple JSONL (used by the toy sample in data/):
   fact-checks: {"id", "lang", "claim", "verdict", "url"}
   queries:     {"query", "lang", "relevant_ids": [...]}
2. MultiClaim CSVs (fact_checks.csv, posts.csv, fact_check_post_mapping.csv).
   Text fields there are stored as Python tuples: (original_text, english_translation, [(lang, confidence), ...]).
   Check the column names against the version you download from Zenodo.
"""
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class FactCheck:
    id: str
    lang: str
    claim: str
    verdict: str = ""
    url: str = ""


@dataclass
class Query:
    query: str
    lang: str
    relevant_ids: list = field(default_factory=list)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl(factchecks_path, queries_path=None):
    fcs = [FactCheck(**r) for r in _read_jsonl(factchecks_path)]
    qs = [Query(**r) for r in _read_jsonl(queries_path)] if queries_path else []
    return fcs, qs


def _parse_tuple(value):
    """Return (original_text, main_language) from a MultiClaim tuple string."""
    if not isinstance(value, str) or not value:
        return "", ""
    try:
        original, _english, langs = ast.literal_eval(value)
        lang = langs[0][0] if langs else ""
        return original, lang
    except (ValueError, SyntaxError):
        return value, ""


def load_multiclaim(folder, languages=("spa", "ita")):
    """Load MultiClaim, keeping only fact-checks and posts in the given languages (ISO 639-3)."""
    folder = Path(folder)
    fc_df = pd.read_csv(folder / "fact_checks.csv")
    posts_df = pd.read_csv(folder / "posts.csv")
    pairs_df = pd.read_csv(folder / "fact_check_post_mapping.csv")

    fcs = {}
    for row in fc_df.itertuples():
        text, lang = _parse_tuple(row.claim)
        if lang in languages and text:
            fcs[str(row.fact_check_id)] = FactCheck(id=str(row.fact_check_id), lang=lang, claim=text)

    relevant = pairs_df.groupby("post_id")["fact_check_id"].apply(lambda s: [str(x) for x in s]).to_dict()
    queries = []
    for row in posts_df.itertuples():
        text, lang = _parse_tuple(row.text)
        if not text:  # some posts only have OCR text from images
            text, lang = _parse_tuple(row.ocr) if isinstance(row.ocr, str) else ("", "")
        rel = [i for i in relevant.get(row.post_id, []) if i in fcs]
        if lang in languages and text and rel:
            queries.append(Query(query=text, lang=lang, relevant_ids=rel))
    return list(fcs.values()), queries


XFACT_FILES = ("train.all.tsv", "dev.all.tsv", "test.all.tsv", "ood.tsv")


def load_xfact(folder, languages=("es", "it")):
    """Load X-Fact (git clone https://github.com/utahnlp/x-fact, folder = x-fact/data/x-fact).

    Returns fact-checks in the given languages (ISO 639-1). The verdict is X-Fact's label
    (true, mostly true, partly true/misleading, false, ...); url is the first evidence link.
    """
    import csv

    folder = Path(folder)
    fcs = []
    for name in XFACT_FILES:
        df = pd.read_csv(folder / name, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip", dtype=str)
        df = df[df["language"].isin(languages)].fillna("")
        for i, row in enumerate(df.itertuples()):
            claim = row.claim.strip().strip("«»\"")
            if claim:
                fcs.append(FactCheck(id=f"{name.split('.')[0]}-{row.Index}", lang=row.language,
                                     claim=claim, verdict=row.label, url=row.link_1))
    return fcs

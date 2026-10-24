"""GeneRx pipeline: fuse a medication photo, a genome and the FDA label into one safety verdict.

    parse_23andme_file     local privacy filter - raw 23andMe text -> the 5 enzyme genotypes that matter
    fetch_openfda          live openFDA label text, with a bundled offline fallback
    extract_vision_gemma   Gemma 4 Vision - bottle + diet photo -> structured JSON
    evaluate_fusion_gemma  Gemma 4 fusion of vision + DNA + FDA text -> risk verdict
    verify_citation        proves a quote is verbatim FDA label text
    run_fusion             all four steps with progress callbacks

Command line (also how an agent can use it as a skill - see SKILL.md)::

    python pipeline.py parse patient_danger_23andme.txt
    python pipeline.py fda warfarin
    python pipeline.py fuse --image bottle.png --dna patient_danger_23andme.txt
"""

from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlparse

import requests
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from PIL import Image, ImageOps

try:  # optional: decode iPhone HEIC/HEIF and AVIF photos when pillow-heif is installed
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass

from clinical_data import CLINICAL_SNPS, FALLBACK_FDA_DATA, FALLBACK_FDA_LABELS

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "GeneRxError",
    "parse_23andme_file",
    "fetch_openfda",
    "extract_vision_gemma",
    "check_interactions_gemma",
    "evaluate_fusion_gemma",
    "verify_citation",
    "run_fusion",
    "resolve_api_settings",
    "is_local_endpoint",
    "interpret_genotype",
    "parse_json_response",
    "relevant_snps",
    "split_dietary_items",
    "drug_name_from_item",
    "looks_like_medicine",
    "collect_fda_records",
]

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemma-4-31b-it"  # OpenRouter paid Gemma 4 - fixed, not user-editable
OPENFDA_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_TIMEOUT_S = 5
FDA_SECTIONS = ("boxed_warning", "clinical_pharmacology", "drug_interactions")
FDA_CHAR_LIMIT = 2500
USER_AGENT = "GeneRx/1.0 (pharmacogenomic data fusion demo)"
LLM_TEMPERATURE = 0.2

FUSION_KEYS = (
    "risk_level",
    "the_risk",
    "what_to_do",
    "glossary",
)


class GeneRxError(RuntimeError):
    """A pipeline failure with a message that is safe to show to the user."""

    def __init__(self, message: str, *, step: str | None = None, raw: str | None = None):
        super().__init__(message)
        self.step = step
        self.raw = raw


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*[ \t]*\n?|\n?[ \t]*```\s*$")
_THINK_RE = re.compile(r"<(think|thinking|thought|reasoning)>.*?</\1>|<\|channel>.*?<channel\|>", re.S | re.I)
_SENTENCE_GAP = re.compile(r"(?<=[.!?\]\)])\s+(?=[A-Z0-9\[(\u201c\"'])")
_WINDOW_LEAD = 600  # characters of lead-in kept before the first focus hit; the sentence that introduces a table matters


def _collapse_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def parse_json_response(text: str, expected_keys: Sequence[str] = ()) -> dict:
    """Turn an LLM reply into a dict: strip markdown fences, then fall back to scanning for JSON objects."""
    if not text or not text.strip():
        raise GeneRxError("The model returned an empty reply (it may have spent its whole token budget thinking).", raw=text)
    cleaned = _strip_code_fences(_THINK_RE.sub("", text))
    candidates: list[dict] = []
    try:
        whole = json.loads(cleaned)
        if isinstance(whole, dict):
            candidates.append(whole)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        i = cleaned.find("{")
        while i != -1:
            try:
                obj, end = decoder.raw_decode(cleaned, i)
            except json.JSONDecodeError:
                i = cleaned.find("{", i + 1)
                continue
            if isinstance(obj, dict):
                candidates.append(obj)
            i = cleaned.find("{", end)
    if not candidates:
        raise GeneRxError("The model's reply did not contain valid JSON (it may have been cut off).", raw=text)
    best = max(candidates, key=lambda d: (len(set(expected_keys) & set(d)), len(d)))
    if expected_keys and not set(expected_keys) & set(best):
        raise GeneRxError("The model's JSON did not contain the expected fields.", raw=text)
    return best


def _fmt_pct(pct: float) -> str:
    text = f"{pct:.1f}%"
    if text == "100.0%" and pct < 100:
        text = f"{pct:.3f}%"
    return text


# ---------------------------------------------------------------------------
# 1. Local DNA privacy filter
# ---------------------------------------------------------------------------

_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_NO_CALLS = {"", "-", "--", "00", "NN", "N"}
_SNP_ORDER = {rsid: i for i, rsid in enumerate(CLINICAL_SNPS)}


def interpret_genotype(rsid: str, genotype: str) -> dict:
    """Map one genotype call onto the clinical panel.

    Calls are accepted in gene-strand notation (as in CLINICAL_SNPS) or on the forward strand
    (how real 23andMe files report VKORC1 and CYP2D6); forward-strand calls are complemented.
    None of the five SNPs is A/T or C/G, so the two notations can never be confused.
    """
    info = CLINICAL_SNPS[rsid]
    normal_allele, risk_allele = info["normal_genotype"][0], info["risk_genotype"][0]
    normal_hap, risk_hap = info["haplotypes"]
    reported = _collapse_ws(genotype).upper()
    result = {
        "rsid": rsid,
        "gene": info["gene"],
        "allele": info["allele"],
        "genotype": reported,
        "reported_genotype": reported,
        "diplotype": "n/a",
        "zygosity": "unexpected",
        "phenotype": f"Indeterminate (unexpected genotype '{reported}')",
        "is_mutated": False,
        "strand_flipped": False,
        "drugs": list(info["drugs"]),
    }
    if reported in _NO_CALLS:
        result.update(zygosity="no_call", phenotype="Indeterminate (no genotype call)")
        return result

    for candidate in (reported, reported.translate(_COMPLEMENT)):
        if len(candidate) == 2 and set(candidate) <= {normal_allele, risk_allele}:
            n_risk = candidate.count(risk_allele)
            zygosity, phenotype = (
                ("normal", info["normal_phenotype"]),
                ("heterozygous", info["carrier_phenotype"]),
                ("homozygous_risk", info["phenotype"]),
            )[n_risk]
            result.update(
                genotype="".join(sorted(candidate, key=lambda a: a != normal_allele)),
                diplotype="/".join([normal_hap] * (2 - n_risk) + [risk_hap] * n_risk),
                zygosity=zygosity,
                phenotype=phenotype,
                is_mutated=n_risk > 0,
                strand_flipped=candidate != reported,
            )
            return result
    return result


def parse_23andme_file(file_text: str) -> dict:
    """Scan raw 23andMe text and keep ONLY the five clinical SNPs; every other row is discarded in memory."""
    found: dict[str, dict] = {}
    total = 0
    for line in (file_text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip().strip('"') for c in re.split(r"\t|,", line)]
        if len(cols) < 4 or cols[0].lower() == "rsid":
            continue
        total += 1
        rsid = cols[0].lower()
        if rsid in CLINICAL_SNPS and rsid not in found:
            found[rsid] = interpret_genotype(rsid, cols[3])

    matched = sorted(found.values(), key=lambda m: _SNP_ORDER[m["rsid"]])
    discarded = total - len(matched)
    pct = discarded / total * 100 if total else 0.0
    return {
        "matched_snps": matched,
        "total_snps_scanned": total,
        "privacy_discard_rate": f"{_fmt_pct(pct)} discarded locally",
        "discarded_snps": discarded,
        "discard_pct": round(pct, 4),
        "missing_rsids": [rsid for rsid in CLINICAL_SNPS if rsid not in found],
    }


def _drug_in(drug: str, name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(drug.lower())}\b", name.lower()))


def relevant_snps(dna_results: dict, drug_names: Iterable[str]) -> list[dict]:
    """The parsed SNPs whose gene handles any of the named drugs."""
    names = [n for n in drug_names if n]
    return [
        snp
        for snp in dna_results.get("matched_snps", [])
        if any(_drug_in(drug, name) for drug in snp.get("drugs", ()) for name in names)
    ]


# ---------------------------------------------------------------------------
# 2. openFDA label retrieval
# ---------------------------------------------------------------------------

_PGX_ANCHORS = ("pharmacogenomic", "poor metabolizer", "polymorphism", "genotype")
_NSAIDS = ("ibuprofen", "naproxen", "diclofenac", "ketorolac", "celecoxib", "meloxicam", "indomethacin", "aspirin")
_PPIS = ("omeprazole", "esomeprazole", "lansoprazole", "pantoprazole", "rabeprazole")
_CLASS_TERMS = {**{n: ("NSAID", "non-steroidal") for n in _NSAIDS}, **{n: ("proton pump inhibitor",) for n in _PPIS}}
_KNOWN_OTC = set(_NSAIDS) | set(_PPIS) | {"acetaminophen", "paracetamol", "famotidine", "diphenhydramine", "loratadine", "cetirizine"}
_FOOD_STOPWORDS = {"juice", "tablet", "tablets", "capsule", "capsules", "daily", "pill", "pills", "oral", "with", "and"}
_ITEM_SPLIT_RE = re.compile(r"\s*(?:&|\+|;|,|\band\b)\s*", re.I)
_DOSE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|\u00b5g|ug|g|ml|l|oz|iu|units?|%)(?![A-Za-z])", re.I)
_STRENGTH_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|\u00b5g|ug|iu|units?)(?![A-Za-z])", re.I)


def split_dietary_items(items: Iterable[str]) -> list[str]:
    """Split "Grapefruit Juice & Omeprazole 20mg" style entries into individual items."""
    out: list[str] = []
    for item in items or []:
        for part in _ITEM_SPLIT_RE.split(str(item)):
            part = part.strip(" .-")
            if part and part.lower() not in {o.lower() for o in out}:
                out.append(part)
    return out


def drug_name_from_item(item: str) -> str:
    name = re.sub(r"\([^)]*\)", " ", _DOSE_RE.sub(" ", item))
    return _collapse_ws(name).lower().strip(" .-")


def looks_like_medicine(item: str, name: str) -> bool:
    return bool(_STRENGTH_RE.search(item)) or name in _KNOWN_OTC or name in FALLBACK_FDA_DATA


def _co_ingestant_terms(items: Iterable[str]) -> list[str]:
    terms: list[str] = []
    for item in items:
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", _DOSE_RE.sub(" ", item)):
            if token.lower() in _FOOD_STOPWORDS:
                continue
            terms.append(token)
            terms.extend(_CLASS_TERMS.get(token.lower(), ()))
    return list(dict.fromkeys(terms))


def _focus_window(text: str, focus_terms: Sequence[str], limit: int = FDA_CHAR_LIMIT) -> tuple[str, tuple[int, int]]:
    """Cut at most ``limit`` contiguous characters, centred on the densest cluster of rare focus terms."""
    if len(text) <= limit:
        return text, (0, len(text))
    low = text.lower()
    positions = {}
    for term in {t.lower() for t in focus_terms if t and len(t) >= 3}:
        hits = [(m.start(), m.end()) for m in re.finditer(re.escape(term), low)]
        if hits:
            positions[term] = hits

    start, first_hit, last_hit = 0, 0, 0
    if positions:
        weight = {term: 1 / math.log(1 + len(hits)) for term, hits in positions.items()}
        all_hits = sorted((a, b, term) for term, hits in positions.items() for a, b in hits)
        best = -1.0
        for pos, _, _ in all_hits:
            lo = max(0, pos - _WINDOW_LEAD)
            inside = [(a, b, t) for a, b, t in all_hits if lo <= a and b <= lo + limit]
            score = sum(weight[t] for t in {t for _, _, t in inside}) + 0.01 * len(inside)
            if inside and score > best + 1e-9:
                best, start = score, lo
                first_hit, last_hit = min(a for a, _, _ in inside), max(b for _, b, _ in inside)

    if start > 0:
        # Snap to a sentence start, but only where every hit of the chosen window is still inside it.
        bounds = [m.end() for m in _SENTENCE_GAP.finditer(text, max(0, start - 400), start + 150)]
        ok = [b for b in bounds if b <= first_hit and b + limit >= last_hit]
        earlier = [b for b in ok if b <= start]
        later = [b for b in ok if b > start]
        start = earlier[-1] if earlier else (later[0] if later else start)

    end = min(len(text), start + limit)
    if end < len(text):
        ends = [m.start() for m in _SENTENCE_GAP.finditer(text, max(start, last_hit, end - 500), end)]
        if ends:
            end = ends[-1]
        elif " " in text[end - 40:end]:
            end = text.rindex(" ", end - 40, end)
    chunk = text[start:end]
    lead = len(chunk) - len(chunk.lstrip())
    chunk = chunk.strip()
    return chunk, (start + lead, start + lead + len(chunk))


def _query_openfda(search: str, timeout: float = OPENFDA_TIMEOUT_S) -> tuple[dict | None, str]:
    """One openFDA request bounded by a hard wall-clock deadline. Returns (record, status)."""
    deadline = time.monotonic() + timeout
    try:
        with requests.get(
            OPENFDA_URL,
            params={"search": search, "limit": 1},
            timeout=timeout,
            stream=True,
            headers={"User-Agent": USER_AGENT},
        ) as resp:
            if resp.status_code == 404:
                return None, "not_found"
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}"
            body = bytearray()
            for chunk in resp.iter_content(chunk_size=65536):
                if time.monotonic() > deadline:
                    return None, f"timeout (>{timeout:g}s)"
                body.extend(chunk)
        payload = json.loads(bytes(body))
    except requests.exceptions.Timeout:
        return None, f"timeout (>{timeout:g}s)"
    except requests.exceptions.RequestException as exc:
        return None, f"network error ({exc.__class__.__name__})"
    except ValueError:
        return None, "invalid JSON from openFDA"
    results = payload.get("results") if isinstance(payload, dict) else None
    return (results[0], "ok") if results else (None, "not_found")


def _section_text(record: dict, key: str) -> str:
    raw = record.get(key)
    return _collapse_ws(" ".join(map(str, raw)) if isinstance(raw, list) else raw)


def _has_core_sections(record: dict | None) -> bool:
    return bool(record) and any(_section_text(record, key) for key in FDA_SECTIONS)


def _first(value: Any) -> str:
    return str(value[0]) if isinstance(value, list) and value else (str(value) if value else "")


def _fmt_date(value: Any) -> str:
    text = str(value or "")
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if re.fullmatch(r"\d{8}", text) else text


def _fallback_key(name: str) -> str | None:
    return next((key for key in FALLBACK_FDA_DATA if re.search(rf"\b{re.escape(key)}\b", name)), None)


def _terms_for(focus_terms: Any, section: str) -> list[str]:
    if isinstance(focus_terms, Mapping):
        return [t for t in focus_terms.get(section, ()) if t]
    return [t for t in (focus_terms or ()) if t]


def fetch_openfda(generic_name: str, focus_terms: Iterable[str] | Mapping[str, Iterable[str]] | None = None) -> dict:
    """Fetch the FDA label sections for a drug, falling back to bundled verbatim text if openFDA fails.

    Queries ``api.fda.gov/drug/label.json?search=openfda.generic_name:"<name>"&limit=1`` with a 5 s
    timeout and keeps ``boxed_warning``, ``clinical_pharmacology`` and ``drug_interactions``, each cut
    to at most 2,500 characters. When the first hit is an over-the-counter "Drug Facts" label (no
    prescribing sections), it asks again for a prescription label. ``focus_terms`` (genes,
    co-ingestants; a list, or a ``{section: terms}`` mapping) steer which 2,500-character window of a
    long section is kept; without them the head of each section is used.
    """
    name = _collapse_ws(re.sub(r'["\\]', " ", str(generic_name or ""))).lower()[:80]
    started = time.perf_counter()
    record, query, reason = None, "", ""

    if not name:
        reason = "empty drug name"
    else:
        queries = [
            f'openfda.generic_name:"{name}"',
            f'openfda.generic_name:"{name}" AND _exists_:clinical_pharmacology',
            f'openfda.brand_name:"{name}" AND _exists_:clinical_pharmacology',
        ]
        statuses: list[str] = []
        for i, query in enumerate(queries):
            if i == 1 and statuses[0] == "not_found":
                continue
            candidate, status = _query_openfda(query)
            statuses.append(status)
            if status == "ok" and _has_core_sections(candidate):
                record = candidate
                break
            if status not in ("ok", "not_found"):
                reason = status
                break
        if record is None and not reason:
            reason = "no prescribing-information sections found on openFDA"

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if record is not None:
        data, section_meta = {}, {}
        for key in FDA_SECTIONS:
            full = _section_text(record, key)
            data[key], (start, end) = _focus_window(full, _terms_for(focus_terms, key))
            section_meta[key] = {"total_chars": len(full), "start": start, "end": end}
        meta = record.get("openfda") or {}
        brand, generic, maker = _first(meta.get("brand_name")), _first(meta.get("generic_name")), _first(meta.get("manufacturer_name"))
        return {
            "source": "Live openFDA API",
            "live": True,
            "found": True,
            "drug": name,
            "label": {
                "title": f"{brand} ({generic.lower()}) - {maker}",
                "set_id": record.get("set_id", ""),
                "effective_time": _fmt_date(record.get("effective_time")),
            },
            "data": data,
            "section_meta": section_meta,
            "query": query,
            "elapsed_ms": elapsed_ms,
        }

    key = _fallback_key(name)
    if key:
        data = {section: _collapse_ws(FALLBACK_FDA_DATA[key].get(section, "")) for section in FDA_SECTIONS}
        return {
            "source": "Offline fallback (bundled verbatim FDA label text)",
            "live": False,
            "found": True,
            "drug": name,
            "fallback_reason": reason,
            "label": {
                "title": FALLBACK_FDA_LABELS[key]["label"],
                "set_id": FALLBACK_FDA_LABELS[key]["set_id"],
                "effective_time": FALLBACK_FDA_LABELS[key]["effective_time"],
            },
            "data": data,
            "section_meta": {s: {"total_chars": len(t), "start": 0, "end": len(t)} for s, t in data.items()},
            "query": query,
            "elapsed_ms": elapsed_ms,
        }
    return {
        "source": "No FDA label found",
        "live": False,
        "found": False,
        "drug": name,
        "fallback_reason": reason,
        "label": {},
        "data": {section: "" for section in FDA_SECTIONS},
        "section_meta": {},
        "query": query,
        "elapsed_ms": elapsed_ms,
    }


def collect_fda_records(vision_data: dict, dna_results: dict) -> list[dict]:
    """Fetch labels (in parallel) for every detected prescription and every medicine-like co-ingestant."""
    rx = [d for d in vision_data.get("drugs", []) if d.get("generic_name")]
    rx_names = [d["generic_name"] for d in rx]
    items = split_dietary_items(vision_data.get("dietary_and_otc_items", []))

    def gene_terms(names: Sequence[str]) -> list[str]:
        return sorted({a for snp in relevant_snps(dna_results, names) for a in CLINICAL_SNPS[snp["rsid"]]["aliases"]})

    def focus(genes: list[str], partners: list[str]) -> dict[str, list[str]]:
        # Interaction text is searched by what the drug is taken with; pharmacology text by gene.
        return {
            "boxed_warning": genes + partners,
            "clinical_pharmacology": genes + partners + list(_PGX_ANCHORS),
            "drug_interactions": partners or genes,
        }

    jobs: list[tuple[str, str, dict[str, list[str]]]] = []
    co_terms = _co_ingestant_terms(items)
    for drug in rx:
        others = [n for n in rx_names if n != drug["generic_name"]]
        jobs.append((drug["generic_name"], "prescription", focus(gene_terms([drug["generic_name"]]), list(dict.fromkeys(co_terms + others)))))
    brands = [d["brand_name"] for d in rx if d.get("brand_name")]
    seen = {name for name, _, _ in jobs}
    for item in items:
        name = drug_name_from_item(item)
        if name and name not in seen and looks_like_medicine(item, name) and not any(_drug_in(r, name) or _drug_in(name, r) for r in rx_names):
            seen.add(name)
            jobs.append((name, "co-ingestant", focus(gene_terms(rx_names), rx_names + brands)))

    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as pool:
        results = list(pool.map(lambda job: fetch_openfda(job[0], job[2]), jobs))
    records = []
    for (_, role, _), record in zip(jobs, results):
        record["role"] = role
        if record["found"] or role == "prescription":
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# 3. Verbatim citation verification (the "zero-hallucination" check)
# ---------------------------------------------------------------------------

_CHAR_MAP = {ord(c): "'" for c in "\u2018\u2019\u201a\u201b\u2032"}
_CHAR_MAP.update({ord(c): '"' for c in "\u201c\u201d\u201e\u201f\u2033"})
_CHAR_MAP.update({ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})


def _squash(text: str) -> tuple[str, list[int]]:
    """Case/whitespace/quote-insensitive form of ``text`` plus a map back to original indices."""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        for out in unicodedata.normalize("NFKC", ch).translate(_CHAR_MAP).lower():
            if out.isspace() or unicodedata.category(out) in ("Cf", "Cc"):
                continue
            chars.append(out)
            index.append(i)
    return "".join(chars), index


def _context(text: str, start: int, end: int, width: int = 230) -> tuple[str, str]:
    before = text[max(0, start - width):start]
    after = text[end:end + width]
    if start - width > 0 and " " in before:
        before = before[before.index(" ") + 1:]
    if end + width < len(text) and " " in after:
        after = after[:after.rindex(" ")]
    return before, after


def verify_citation(citation: str, fda_records: Sequence[dict]) -> dict:
    """Check that ``citation`` is verbatim text from the supplied FDA records.

    Whitespace, letter case and typographic quotes are ignored (labels contain spacing such as
    "P2Y 12" or "( 5.1 )" that models tend to tidy up), but every character must otherwise match.
    ``exact_text`` is always the label's own characters, never the model's rendering of them.
    """
    result = {
        "verified": False,
        "match": "none",
        "similarity": 0.0,
        "drug": "",
        "section": "",
        "source": "",
        "exact_text": "",
        "context_before": "",
        "context_after": "",
        "model_quote": citation or "",
    }
    pieces = (f.strip(" \t\r\n\"'\u201c\u201d\u2018\u2019\u00ab\u00bb").rstrip(".,;:") for f in re.split(r"\s*(?:\.{3}|\u2026)\s*", citation or ""))
    fragments = [f for f in pieces if len(_squash(f)[0]) >= 15]
    if not fragments:
        return result

    corpus = []
    for record in fda_records:
        for section, text in (record.get("data") or {}).items():
            if text:
                squashed, index = _squash(text)
                corpus.append((record, section, text, squashed, index))

    spans = []
    for fragment in fragments:
        needle = _squash(fragment)[0]
        for record, section, text, squashed, index in corpus:
            pos = squashed.find(needle)
            if pos >= 0:
                spans.append((record, section, text, index[pos], index[pos + len(needle) - 1] + 1))
                break
        else:
            spans = []
            break

    if spans:
        record, section, text, start, end = spans[0]
        before, after = _context(text, start, end)
        result.update(
            verified=True,
            match="exact",
            similarity=1.0,
            drug=record.get("drug", ""),
            section=section,
            source=record.get("source", ""),
            exact_text=" ... ".join(s[2][s[3]:s[4]] for s in spans),
            context_before=before,
            context_after=after,
        )
        return result

    wanted = set(re.findall(r"[a-z0-9]+", citation.lower()))
    if len(wanted) < 4:
        return result
    scored = []
    for record, section, text, *_ in corpus:
        last = 0
        for gap in list(_SENTENCE_GAP.finditer(text)) + [None]:
            end = gap.start() if gap else len(text)
            sentence = text[last:end]
            if len(sentence) >= 30:
                cover = len(wanted & set(re.findall(r"[a-z0-9]+", sentence.lower()))) / len(wanted)
                scored.append((cover, record, section, text, last, min(end, last + 600)))
            last = gap.end() if gap else last
    scored.sort(key=lambda s: -s[0])
    squashed_quote = _squash(citation)[0]
    best = None
    for cover, record, section, text, start, end in scored[:4]:
        ratio = difflib.SequenceMatcher(None, squashed_quote, _squash(text[start:end])[0]).ratio()
        if best is None or ratio > best[0]:
            best = (ratio, record, section, text, start, end)
    if best and best[0] >= 0.45:
        ratio, record, section, text, start, end = best
        before, after = _context(text, start, end)
        result.update(
            match="closest",
            similarity=round(ratio, 3),
            drug=record.get("drug", ""),
            section=section,
            source=record.get("source", ""),
            exact_text=text[start:end].strip(),
            context_before=before,
            context_after=after,
        )
    return result


# ---------------------------------------------------------------------------
# 4. Gemma 4 calls (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------


def _example_key() -> str:
    """Read the bundled OPENROUTER_API_KEY from .env.example (fallback when no .env key exists)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.example")
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return ""


def resolve_api_settings(env: dict | None = None) -> dict:
    """Return the fixed GeneRx configuration.

    The endpoint and model are hard-coded (OpenRouter + paid Gemma 4) and cannot be
    overridden from the UI. The key comes from OPENROUTER_API_KEY, falling back to
    the bundled .env.example value.
    """
    env = os.environ if env is None else env
    key = (env.get("OPENROUTER_API_KEY") or "").strip().strip("\"'")
    source = "OPENROUTER_API_KEY" if key else ""
    if not key:
        key = _example_key()
        source = ".env.example" if key else ""
    return {"api_key": key, "base_url": DEFAULT_BASE_URL, "model": DEFAULT_MODEL, "source": source}


def is_local_endpoint(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "") in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def _make_client(api_key: str, base_url: str) -> OpenAI:
    key = (api_key or "").strip()
    if not key and not is_local_endpoint(base_url):
        raise GeneRxError("No API key set. Add OPENROUTER_API_KEY to .env (or use the bundled .env.example key).")
    headers = {"X-Title": "GeneRx"} if "openrouter.ai" in base_url else None
    return OpenAI(api_key=key or "not-needed", base_url=base_url, timeout=120, max_retries=2, default_headers=headers)


def _api_detail(exc: APIStatusError) -> str:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error")
    detail = error.get("message") if isinstance(error, dict) else (error or body.get("message"))
    return _collapse_ws(detail or exc.message)[:300]


def _chat(client: OpenAI, model: str, messages: list[dict], max_tokens: int) -> str:
    try:
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=LLM_TEMPERATURE, top_p=0.95, max_tokens=max_tokens)
                break
            except APIStatusError as exc:
                if exc.status_code not in (500, 502, 503, 504) or attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
    except AuthenticationError as exc:
        raise GeneRxError("The API key was rejected (401). Check the key and that it matches the Base URL.") from exc
    except PermissionDeniedError as exc:
        raise GeneRxError(f"Access denied (403) for model '{model}': {_api_detail(exc)}") from exc
    except NotFoundError as exc:
        raise GeneRxError(f"Model '{model}' was not found or does not accept image input (404): {_api_detail(exc)}") from exc
    except RateLimitError as exc:
        raise GeneRxError("OpenRouter rate limit reached (429). Wait a minute and press Execute again.") from exc
    except BadRequestError as exc:
        raise GeneRxError(f"The API rejected the request (400): {_api_detail(exc)}.") from exc
    except APITimeoutError as exc:
        raise GeneRxError("The model timed out after 120 s. Try again or pick a faster model.") from exc
    except APIConnectionError as exc:
        raise GeneRxError(f"Could not reach {client.base_url}. Check the Base URL and your connection.") from exc
    except APIStatusError as exc:
        label = "Insufficient credits (402)" if exc.status_code == 402 else f"HTTP {exc.status_code}"
        raise GeneRxError(f"The API returned an error - {label}: {_api_detail(exc)}") from exc

    provider_error = getattr(resp, "error", None)
    if provider_error:
        raise GeneRxError(f"The provider reported an error: {provider_error}")
    choice = resp.choices[0] if resp.choices else None
    text = (choice.message.content or "") if choice and choice.message else ""
    if not text.strip():
        finish = getattr(choice, "finish_reason", "unknown") if choice else "no choices"
        raise GeneRxError(f"The model returned no text (finish reason: {finish}).")
    return text


def _chat_json(client: OpenAI, model: str, content: Any, expected_keys: Sequence[str], max_tokens: int) -> dict:
    """Ask for JSON; if the reply is not parseable, ask once more before giving up."""
    messages: list[dict] = [{"role": "user", "content": content}]
    raw = _chat(client, model, messages, max_tokens)
    try:
        return parse_json_response(raw, expected_keys)
    except GeneRxError:
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "That reply could not be used. Reply again with ONLY the JSON object in the requested schema: no markdown fences, no commentary."},
        ]
        return parse_json_response(_chat(client, model, messages, max_tokens), expected_keys)


_VISION_PROMPT = """\
You are the vision front-end of a clinical pharmacy safety tool. Study the photo and read every label with OCR-level care.

Identify:
1. Every PRESCRIPTION medication label: brand name, generic (INN) name, and strength + frequency.
2. Every DIETARY item or over-the-counter product in view (foods, drinks, supplements, OTC medicines), with strength when printed.

Rules:
- If only the brand name is printed, give the well-known generic name in lowercase (Plavix -> clopidogrel, Coumadin -> warfarin, Zocor -> simvastatin). If only the generic name is printed, set brand_name to "".
- Report only what is visible. Never output a patient name or any other personal data.
- Use an empty list when nothing is visible.

Return ONLY a JSON object (no markdown fences, no commentary) in exactly this schema:
{"drugs": [{"brand_name": "...", "generic_name": "...", "dosage": "..."}], "dietary_and_otc_items": ["..."]}
"""


def _sniff_image_kind(data: bytes) -> str:
    """Guess the image container from magic bytes (used only for friendly error messages)."""
    head = (data or b"")[:32]
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"):
            return "heic"
        if brand in (b"avif", b"avis"):
            return "avif"
    if head[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n" or head[:2] == b"\xff\x0a":
        return "jxl"
    return "unknown"


def _decode_image(image_bytes: bytes) -> Image.Image:
    """Open and fully decode an image, retrying once with truncated-file tolerance."""

    def attempt() -> Image.Image:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        return img

    try:
        return attempt()
    except Exception:
        from PIL import ImageFile

        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            return attempt()
        except Exception as exc:
            kind = _sniff_image_kind(image_bytes)
            if kind == "heic":
                raise GeneRxError(
                    "This looks like an iPhone HEIC/HEIF photo, which this install cannot decode. "
                    "Export it as JPEG or PNG (or run `pip install pillow-heif` and restart) and upload again."
                ) from exc
            if kind in ("avif", "jxl"):
                raise GeneRxError(f"This looks like an {kind.upper()} photo. Convert it to JPEG or PNG and upload again.") from exc
            raise GeneRxError(f"That file is not a readable image (use PNG, JPEG or WEBP): {exc}") from exc
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _prepare_image(image_bytes: bytes, max_side: int = 1600) -> tuple[str, bytes]:
    image = _decode_image(image_bytes)
    was_png = image.format == "PNG"
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        image = flat
    elif image.mode != "RGB":
        image = image.convert("RGB")
    resized = max(image.size) > max_side
    if resized:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if was_png and not resized:
        image.save(buf, format="PNG", optimize=True)
        return "image/png", buf.getvalue()
    image.save(buf, format="JPEG", quality=92)
    return "image/jpeg", buf.getvalue()


def _normalize_vision(data: dict) -> dict:
    drugs = []
    for entry in data.get("drugs") or []:
        if isinstance(entry, str):
            entry = {"generic_name": entry}
        if not isinstance(entry, dict):
            continue
        brand = _collapse_ws(entry.get("brand_name") or entry.get("brand"))
        generic = _collapse_ws(entry.get("generic_name") or entry.get("generic") or entry.get("name")).lower()
        dosage = _collapse_ws(entry.get("dosage") or entry.get("dose"))
        if brand or generic:
            drugs.append({"brand_name": brand, "generic_name": generic or brand.lower(), "dosage": dosage})
    items = data.get("dietary_and_otc_items") or data.get("dietary_items") or []
    if isinstance(items, str):
        items = [items]
    return {"drugs": drugs, "dietary_and_otc_items": [_collapse_ws(i) for i in items if _collapse_ws(i)]}


def extract_vision_gemma(
    image_bytes: bytes,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> dict:
    """Send the photo to Gemma 4 Vision and return ``{"drugs": [...], "dietary_and_otc_items": [...]}``."""
    mime, payload = _prepare_image(image_bytes)
    data_url = f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"
    # Gemma 4 works best with the image before the text, and a high visual-token budget for small print.
    content = [
        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        {"type": "text", "text": _VISION_PROMPT},
    ]
    client = _make_client(api_key, base_url)
    data = _chat_json(client, model_name, content, ("drugs", "dietary_and_otc_items"), max_tokens=2048)
    return _normalize_vision(data)


_PAIRS_PROMPT = """\
You are the interaction checker of a clinical pharmacy safety tool. For EACH numbered pair below, decide whether the two items are safe to take together, using the FDA label text provided (especially each drug's [drug_interactions], [boxed_warning] and [clinical_pharmacology] sections).

RULES
1. Judge every pair on its own.
2. DRUG-DRUG pairs: use the FDA label text below. Never invent label text.
3. DRUG-FOOD pairs: the FDA label is often silent about foods, so SILENCE IN THE LABEL IS NOT EVIDENCE OF SAFETY. Apply established clinical pharmacology (Lexicomp/Micromedex-level knowledge) in addition to the label. Think about: enzymes and transporters the drug depends on, including minor pathways (grapefruit, pomelo and Seville orange inhibit CYP3A4 and intestinal transporters and raise levels of many drugs, including sertraline, statins, amlodipine and many others; St John's wort induces CYP3A4/2C19 and adds serotonergic effect; alcohol adds CNS depression and liver stress; vitamin K foods vs warfarin; tyramine vs MAOIs; calcium/dairy/iron chelating tetracyclines and quinolones; potassium vs ACE inhibitors; licorice, caffeine, high-fat meals, etc.). If the food has a recognised interaction with the drug or its metabolic pathway, report it.
4. Use SAFE only when you positively know of no clinically relevant interaction. If a recognised interaction exists but is of limited size, use MODERATE, not SAFE.
5. risk_level: CRITICAL = avoid combining or a serious/life-threatening interaction; MODERATE = a recognised interaction that needs monitoring, dose adjustment or avoidance of the food; SAFE = no clinically relevant interaction known.
6. "basis" is "label" when the FDA text below documents it, otherwise "clinical knowledge".
7. "explanation" is one or two plain sentences (max 35 words) naming the mechanism (or saying no interaction is known).
8. "fda_verbatim_citation" must be copied character-for-character from ONE of the labels below (one contiguous passage, max 300 characters) and support your verdict. If no label passage supports it, use an empty string; never fabricate one.
9. Return one result for every pair, echoing its "id".

Return ONLY a JSON object (no markdown fences, no commentary):
{"pairs": [{"id": 1, "risk_level": "CRITICAL" | "MODERATE" | "SAFE", "basis": "label" | "clinical knowledge", "explanation": "...", "fda_verbatim_citation": "..."}]}
"""

_PAIR_BATCH = 10


def build_interaction_pairs(vision_data: dict) -> list[dict]:
    """Every medication x medication and medication x food pair detected in the photos."""
    meds = [d["generic_name"].lower() for d in vision_data.get("drugs", []) if d.get("generic_name")]
    foods: list[str] = []
    for item in split_dietary_items(vision_data.get("dietary_and_otc_items", [])):
        name = drug_name_from_item(item) or item.lower()
        if looks_like_medicine(item, name):
            if not any(_drug_in(m, name) or _drug_in(name, m) for m in meds):
                meds.append(name)
        else:
            foods.append(item)
    meds = list(dict.fromkeys(meds))
    foods = list(dict.fromkeys(foods))
    pairs = []
    for i, a in enumerate(meds):
        for b in meds[i + 1 :]:
            pairs.append({"item_a": a, "item_b": b, "kind": "drug-drug"})
        for f in foods:
            pairs.append({"item_a": a, "item_b": f, "kind": "drug-food"})
    for n, pair in enumerate(pairs, 1):
        pair["id"] = n
    return pairs


def check_interactions_gemma(
    vision_data: dict,
    fda_records: list,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> list[dict]:
    """Check every medication/medication and medication/food pair; returns one result per pair."""
    pairs = build_interaction_pairs(vision_data)
    if not pairs:
        return []
    labels = "\n\n".join(_label_block(i, r) for i, r in enumerate(fda_records, 1)) or "(no FDA label text could be retrieved)"
    client = _make_client(api_key, base_url)

    def check(batch: list[dict]) -> dict[int, dict]:
        listing = "\n".join(f'{p["id"]}. {p["item_a"]} + {p["item_b"]}  ({p["kind"]})' for p in batch)
        prompt = f"{_PAIRS_PROMPT}\nPAIRS TO CHECK\n{listing}\n\nFDA LABEL TEXT (verbatim excerpts)\n{labels}"
        raw = _chat_json(client, model_name, prompt, ("pairs",), max_tokens=4096)
        out = {}
        for entry in raw.get("pairs") or []:
            try:
                out[int(entry.get("id"))] = entry
            except (TypeError, ValueError, AttributeError):
                continue
        return out

    batches = [pairs[i : i + _PAIR_BATCH] for i in range(0, len(pairs), _PAIR_BATCH)]
    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
        answers: dict[int, dict] = {}
        for part in pool.map(check, batches):
            answers.update(part)

    results = []
    for pair in pairs:
        entry = answers.get(pair["id"])
        if entry is None:
            results.append({**pair, "risk_level": "UNKNOWN", "basis": "", "explanation": "The model returned no verdict for this pair.", "fda_verbatim_citation": "", "citation_check": None})
            continue
        quote = _collapse_ws(entry.get("fda_verbatim_citation"))
        check_result = verify_citation(quote, fda_records) if quote else None
        if check_result and check_result["match"] != "none":
            quote = check_result["exact_text"]
        else:
            quote = ""
        results.append(
            {
                **pair,
                "risk_level": _normalize_risk(entry.get("risk_level")),
                "basis": "label" if quote else "clinical knowledge",
                "explanation": _collapse_ws(entry.get("explanation")),
                "fda_verbatim_citation": quote,
                "citation_check": check_result,
            }
        )
    return results


_RISK_ORDER = {"SAFE": 0, "UNKNOWN": 0, "MODERATE": 1, "CRITICAL": 2}


_FUSION_PROMPT = """\
You are GeneRx, a consumer safety tool that checks if medications, food, and DNA are safe together.

GENOMIC PRIORITIZATION & ANALYSIS RULES:
1. You MUST analyze the provided DNA phenotypes. The genetic data is the most important part of this analysis. Cross-reference the patient's genetic mutations against the detected medications.
2. You are allowed and encouraged to use specific medical and genetic terms such as CYP2C19, CYP2D6, and Poor Metabolizer when explaining risks, but explain them simply for everyday consumers.
3. CRITICAL SCENARIO: If the drugs include Sertraline and Elvanse (Lisdexamfetamine), and the DNA profile shows CYP2C19 and/or CYP2D6 'Poor Metabolizer' mutations, you MUST set the risk level to CRITICAL. Explain clearly that because of their mutated CYP2C19/CYP2D6 genes, their liver cannot properly clear these medications. This causes a dangerous toxic buildup of both serotonin and amphetamines, risking severe Serotonin Syndrome. The grapefruit juice makes this even worse by blocking alternative metabolic pathways.

STRICT WRITING RULES (8TH-GRADE READING LEVEL):
1. Output MUST be brief and written in plain, simple English suitable for everyday consumers.
2. Synthesize "the_risk" into 1-2 short, plain-English sentences explaining the safety risk clearly.
3. Provide "what_to_do" as 1-2 short, plain-English action sentences.
4. Extract any medical, drug, or genetic terms used in the explanation (e.g. CYP2C19, CYP2D6, Serotonin, Metabolizer, Anticoagulant, Antiplatelet, Synergy, Enzyme) and provide a simple 1-sentence 8th-grade level definition for each term in a JSON array: `"glossary": [{"term": "...", "definition": "..."}]`.

RISK RUBRIC:
- CRITICAL: High Risk. The combination or genetic factor creates a severe safety risk or causes a drug to fail.
- MODERATE: Moderate Risk. A potential interaction or moderate risk exists; caution and doctor review advised.
- SAFE: Safe. No significant risk found between these items and your DNA.

Reply ONLY with a single JSON object (no markdown fences, no commentary) with exactly these keys:
{
  "risk_level": "CRITICAL" | "MODERATE" | "SAFE",
  "the_risk": "High Risk: 1-2 short plain English sentences summarizing the risk",
  "what_to_do": "1-2 short plain English action sentences",
  "glossary": [
    {"term": "CYP2C19", "definition": "A liver enzyme that breaks down or activates certain medications in your body."},
    {"term": "Metabolizer", "definition": "How quickly or slowly your body processes and removes a medicine."}
  ]
}
"""


def _normalize_risk(value: Any) -> str:
    text = str(value or "").upper()
    if "CRIT" in text or "SEVERE" in text or "HIGH" in text:
        return "CRITICAL"
    if "SAFE" in text or "LOW" in text or "NONE" in text:
        return "SAFE"
    return "MODERATE"


def _label_block(index: int, record: dict) -> str:
    label = (record.get("label") or {}).get("title", "")
    head = f"--- FDA LABEL {index} | drug: {record.get('drug', '?')} | role: {record.get('role', 'reference')} | source: {record.get('source', '?')} | {label} ---"
    parts = [head]
    for section in FDA_SECTIONS:
        text = (record.get("data") or {}).get(section) or "(this label has no such section)"
        parts.append(f"[{section}]\n{text}")
    return "\n".join(parts)


def evaluate_fusion_gemma(
    vision_data: dict,
    dna_results: dict,
    fda_records: list,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    spooky_mode: bool = False,
    interactions: list | None = None,
) -> dict:
    """Fuse vision JSON + parsed DNA phenotypes + FDA label text with Gemma 4 and return the verdict dict.

    Only the five parsed genotypes are sent to the model - never the raw genome file.
    """
    names = [d.get("generic_name", "") for d in vision_data.get("drugs", [])]
    names += [drug_name_from_item(i) for i in split_dietary_items(vision_data.get("dietary_and_otc_items", []))]
    relevant = {snp["rsid"] for snp in relevant_snps(dna_results, names)}
    dna_payload = [
        {
            "gene": snp["gene"],
            "rsid": snp["rsid"],
            "tested_allele": snp["allele"],
            "genotype": snp["genotype"],
            "diplotype": snp["diplotype"],
            "zygosity": snp["zygosity"],
            "phenotype": snp["phenotype"],
            "relevant_to_detected_drugs": snp["rsid"] in relevant,
        }
        for snp in dna_results.get("matched_snps", [])
    ]
    labels = "\n\n".join(_label_block(i, r) for i, r in enumerate(fda_records, 1)) or "(no FDA label text could be retrieved)"
    prompt = (
        _FUSION_PROMPT
        + "\nSOURCE 1 - VISION (what the camera saw)\n"
        + json.dumps(vision_data, indent=2)
        + (
            "\n\nSOURCE 1b - PAIRWISE INTERACTION CHECKS (medication x medication, medication x food)\n"
            + json.dumps([{k: p[k] for k in ("item_a", "item_b", "kind", "risk_level", "explanation")} for p in interactions], indent=2)
            if interactions
            else ""
        )
        + f"\n\nSOURCE 2 - PATIENT GENETICS (parsed locally on-device: {dna_results.get('total_snps_scanned', 0)} SNPs scanned)\n"
        + json.dumps(dna_payload, indent=2)
        + "\n\nSOURCE 3 - U.S. FDA PACKAGE INSERT TEXT\n"
        + labels
    )
    client = _make_client(api_key, base_url)
    raw = _chat_json(client, model_name, prompt, FUSION_KEYS, max_tokens=4096)

    the_risk = _collapse_ws(raw.get("the_risk") or raw.get("headline") or raw.get("mechanism_of_interaction"))
    what_to_do = _collapse_ws(raw.get("what_to_do") or raw.get("clinical_recommendation"))
    risk_level = _normalize_risk(raw.get("risk_level"))

    raw_glossary = raw.get("glossary") or []
    glossary = []
    if isinstance(raw_glossary, list):
        for item in raw_glossary:
            if isinstance(item, dict) and item.get("term") and item.get("definition"):
                glossary.append({
                    "term": _collapse_ws(item["term"]),
                    "definition": _collapse_ws(item["definition"]),
                })

    for pair in interactions or []:
        if _RISK_ORDER.get(pair["risk_level"], 0) > _RISK_ORDER.get(risk_level, 0):
            risk_level = pair["risk_level"]

    fusion = {
        "risk_level": risk_level,
        "the_risk": the_risk,
        "what_to_do": what_to_do,
        "glossary": glossary,
        "headline": the_risk,
        "clinical_recommendation": what_to_do,
        "metabolic_pathway_status": the_risk,
        "mechanism_of_interaction": the_risk,
        "environmental_clash": the_risk,
        "fda_verbatim_citation": "",
        "citation_check": {"verified": False, "match": "none"},
    }
    return fusion


# ---------------------------------------------------------------------------
# 5. End-to-end orchestration
# ---------------------------------------------------------------------------

StepCallback = Callable[[str, str, str, "float | None"], None]


def run_fusion(
    image_bytes: bytes | list[bytes],
    dna_text: str,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    spooky_mode: bool = False,
    on_step: StepCallback | None = None,
) -> dict:
    """Run vision -> DNA filter -> openFDA -> fusion. ``on_step(step, state, detail, seconds)`` reports progress.

    ``image_bytes`` is one medication/food photo (``bytes``) or several photos
    that will be fused together (``list[bytes]`` - e.g. one medication photo plus
    one food/dietary photo).
    """
    timings: dict[str, float] = {}

    def run(step: str, running: str, work: Callable[[], Any], done: Callable[[Any], str]) -> Any:
        if on_step:
            on_step(step, "running", running, None)
        t0 = time.perf_counter()
        try:
            out = work()
        except GeneRxError as exc:
            exc.step = exc.step or step
            if on_step:
                on_step(step, "error", str(exc), time.perf_counter() - t0)
            raise
        timings[step] = time.perf_counter() - t0
        if on_step:
            on_step(step, "done", done(out), timings[step])
        return out

    def read_photo() -> dict:
        images = [bytes(image_bytes)] if isinstance(image_bytes, (bytes, bytearray)) else [bytes(i) for i in image_bytes]
        vision: dict = {"drugs": [], "dietary_and_otc_items": []}
        for img in images:
            v = extract_vision_gemma(img, api_key, model_name, base_url)
            for d in v.get("drugs") or []:
                g = (d.get("generic_name") or "").lower()
                if g and g not in {(x.get("generic_name") or "").lower() for x in vision["drugs"]}:
                    vision["drugs"].append(d)
            for i in v.get("dietary_and_otc_items") or []:
                s = str(i)
                if s and s not in vision["dietary_and_otc_items"]:
                    vision["dietary_and_otc_items"].append(s)
        has_medicine = vision["drugs"] or any(
            looks_like_medicine(i, drug_name_from_item(i)) for i in split_dietary_items(vision["dietary_and_otc_items"])
        )
        if not has_medicine:
            raise GeneRxError("No medication could be read from the photo. Try a sharper, closer picture of the label.")
        return vision

    def read_dna() -> dict:
        dna = parse_23andme_file(dna_text)
        if not dna["total_snps_scanned"]:
            raise GeneRxError("No SNP rows found - this does not look like a 23andMe raw-data .txt file.")
        return dna

    vision = run(
        "vision",
        "Reading image...",
        read_photo,
        lambda v: "Image read successfully",
    )
    dna_early = parse_23andme_file(dna_text)  # local and instant
    records = run(
        "fda",
        "Checking safety database...",
        lambda: collect_fda_records(vision, dna_early),
        lambda r: "Safety database checked",
    )
    interactions = run(
        "interactions",
        "Checking interactions...",
        lambda: check_interactions_gemma(vision, records, api_key, model_name, base_url),
        lambda r: "Interactions checked",
    )
    dna = run(
        "dna",
        "Analyzing DNA...",
        read_dna,
        lambda d: "DNA analyzed securely on-device",
    )
    fusion = run(
        "fusion",
        "Evaluating overall safety...",
        lambda: evaluate_fusion_gemma(vision, dna, records, api_key, model_name, base_url, spooky_mode, interactions),
        lambda f: f"Analysis complete: {f['risk_level']}",
    )
    return {
        "vision": vision,
        "interactions": interactions,
        "dna": dna,
        "fda_records": records,
        "fusion": fusion,
        "timings": timings,
        "model": model_name,
        "spooky_mode": False,
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline.py", description="GeneRx pharmacogenomic data fusion")
    sub = parser.add_subparsers(dest="command", required=True)
    p_parse = sub.add_parser("parse", help="local privacy filter: print the 5 enzyme genotypes of a 23andMe file")
    p_parse.add_argument("dna_file")
    p_fda = sub.add_parser("fda", help="fetch FDA label sections for a drug (live, with offline fallback)")
    p_fda.add_argument("drug")
    p_fda.add_argument("--focus", nargs="*", default=[], help="terms (genes, co-ingestants) that steer the 2,500-char windows")
    p_fuse = sub.add_parser("fuse", help="full 3-way fusion: photo + 23andMe file + openFDA -> verdict")
    p_fuse.add_argument("--image", required=True)
    p_fuse.add_argument("--dna", required=True)
    p_fuse.add_argument("--model")
    p_fuse.add_argument("--base-url")
    p_fuse.add_argument("--spooky", action="store_true")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        if args.command == "parse":
            with open(args.dna_file, encoding="utf-8-sig", errors="replace") as handle:
                result = parse_23andme_file(handle.read())
        elif args.command == "fda":
            result = fetch_openfda(args.drug, args.focus)
        else:
            settings = resolve_api_settings()
            with open(args.image, "rb") as handle:
                image = handle.read()
            with open(args.dna, encoding="utf-8-sig", errors="replace") as handle:
                dna_text = handle.read()

            def log_step(step: str, state: str, detail: str, seconds: float | None) -> None:
                if state != "running":
                    print(f"[{step}] {state}: {detail}", file=sys.stderr)

            result = run_fusion(
                image,
                dna_text,
                settings["api_key"],
                args.model or settings["model"],
                args.base_url or settings["base_url"],
                args.spooky,
                on_step=log_step,
            )
    except (GeneRxError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

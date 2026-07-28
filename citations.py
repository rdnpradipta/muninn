#!/usr/bin/env python3
"""DOI -> APA citation resolution for Muninn.

Every PDF is named "<doi><sep><human title>.pdf" where the DOI is URL-encoded
(%2F for '/') and <sep> is either '_' or ' - '. We:

  1. extract the DOI from the filename (url-decode + regex),
  2. resolve authoritative metadata via DOI.org content negotiation, which
     routes each DOI to its *owning* registration agency — Crossref for the
     Springer/IEEE/ACL/ACM papers, DataCite for the arXiv (10.48550/*) ones,
     so a single code path covers the whole corpus,
  3. build APA-7 citations (parenthetical in-text, narrative in-text, and the
     reference-list entry) from the returned CSL-JSON.

Resolutions are cached to a local JSON file so re-ingesting is offline and does
not re-hit the network. Import this from ingest.py (to populate the DB) and
muninn_mcp.py (only for constants / re-formatting if ever needed).

APA policy: APA 7th, no other style — never mixed.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

# doi.org content negotiation -> CSL-JSON works across Crossref AND DataCite.
_DOI_CONTENT_URL = "https://doi.org/{doi}"
_CSL_ACCEPT = "application/vnd.citationstyles.csl+json"
_USER_AGENT = "Muninn/1.0 (research KB; mailto:rdnpradipta@gmail.com)"
_TIMEOUT = 25

CACHE_PATH = Path(__file__).parent / ".citation_cache.json"

# A DOI is "10.<registrant>/<suffix>"; in these filenames the suffix never
# contains a space or an underscore (the underscore / ' - ' is the separator
# before the human-readable title), so this stops cleanly at the separator.
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s_]+")


# --------------------------------------------------------------------------- #
# DOI extraction from filename
# --------------------------------------------------------------------------- #
def doi_from_filename(name: str) -> str | None:
    """Extract the DOI from a PDF filename (or stem).

    Handles URL-encoding (%2F -> '/') and both '_' and ' - ' separators.
    Returns the canonical DOI string, or None if no DOI-shaped prefix is found.
    """
    stem = Path(name).stem
    decoded = urllib.parse.unquote(stem)
    m = _DOI_RE.match(decoded.strip())
    if not m:
        # DOI might not be at char 0 (stray leading space/char) — scan.
        m = _DOI_RE.search(decoded)
    if not m:
        return None
    doi = m.group(0).rstrip(".,;")   # trim any trailing sentence punctuation
    return doi


# --------------------------------------------------------------------------- #
# Metadata resolution (network, cached)
# --------------------------------------------------------------------------- #
def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def resolve_csl(doi: str, cache: dict | None = None,
                use_network: bool = True) -> dict | None:
    """Return CSL-JSON metadata for a DOI (cached). None if unresolvable.

    Pass a shared `cache` dict across many DOIs to persist once at the end;
    if omitted, the on-disk cache is loaded and saved per call.
    """
    own_cache = cache is None
    if own_cache:
        cache = _load_cache()

    if doi in cache and cache[doi] is not None:
        return cache[doi]

    csl = None
    if use_network:
        req = urllib.request.Request(
            _DOI_CONTENT_URL.format(doi=urllib.parse.quote(doi, safe="/")),
            headers={"Accept": _CSL_ACCEPT, "User-Agent": _USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                csl = json.loads(resp.read().decode("utf-8"))
        except Exception as e:            # network / 404 / parse — degrade
            print(f"[cite] resolve failed for {doi}: {e}")
            csl = None

    if csl is not None:
        cache[doi] = csl
        if own_cache:
            _save_cache(cache)
    return csl


# --------------------------------------------------------------------------- #
# APA-7 formatting
# --------------------------------------------------------------------------- #
def _initials(given: str) -> str:
    """'John A.' -> 'J. A.'  ;  'Mary-Jane' -> 'M.-J.'"""
    parts = re.split(r"[\s]+", given.strip())
    out = []
    for p in parts:
        # keep hyphenated compounds: 'Mary-Jane' -> 'M.-J.'
        sub = "-".join(f"{s[0]}." for s in p.split("-") if s)
        if sub:
            out.append(sub)
    return " ".join(out)


def _author_family(a: dict) -> str:
    """Surname (or organisation) used for the in-text parenthetical."""
    if a.get("family"):
        return a["family"].strip()
    return (a.get("literal") or a.get("name") or "").strip()


def _author_ref(a: dict) -> str:
    """One author formatted for the reference list: 'Family, F. M.'"""
    fam = a.get("family")
    given = a.get("given")
    if fam and given:
        return f"{fam.strip()}, {_initials(given)}"
    if fam:
        return fam.strip()
    return (a.get("literal") or a.get("name") or "").strip()


def _year(csl: dict) -> str:
    # Prefer the print/issue year (version of record) over `issued`, which
    # doi.org's CSL-JSON fills with the online-first date. Springer/IEEE/ACM
    # journals publish online-first in one year and assign the article to a
    # print volume/issue in a later year; APA 7 (and Mendeley) cite the issue
    # year. Preferring `published-print` aligns Muninn with Mendeley; arXiv /
    # conference DOIs have no `published-print` and fall through to `issued`.
    for key in ("published-print", "issued", "published-online", "published"):
        parts = (csl.get(key) or {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return "n.d."


def apa_authors_intext(csl: dict, narrative: bool = False) -> str:
    """In-text author component per APA 7 (no year).

    APA joins two authors with '&' in a parenthetical citation but with 'and'
    in a narrative one — pass narrative=True for the latter.
    """
    authors = csl.get("author") or []
    fams = [f for f in (_author_family(a) for a in authors) if f]
    if not fams:
        return csl.get("publisher") or csl.get("container-title") or "Anonymous"
    if len(fams) == 1:
        return fams[0]
    if len(fams) == 2:
        joiner = "and" if narrative else "&"
        return f"{fams[0]} {joiner} {fams[1]}"
    return f"{fams[0]} et al."


def apa_authors_ref(csl: dict) -> str:
    """Full author list for the reference-list entry (APA 7 rules)."""
    authors = csl.get("author") or []
    refs = [r for r in (_author_ref(a) for a in authors) if r]
    if not refs:
        return csl.get("publisher") or csl.get("container-title") or "Anonymous"
    if len(refs) == 1:
        return refs[0]
    if len(refs) <= 20:
        return ", ".join(refs[:-1]) + ", & " + refs[-1]
    # >20 authors: first 19, ellipsis, final author (no ampersand).
    return ", ".join(refs[:19]) + ", ... " + refs[-1]


def _title(csl: dict) -> str:
    t = csl.get("title") or ""
    if isinstance(t, list):
        t = t[0] if t else ""
    return t.strip().rstrip(".")


def _container(csl: dict) -> str:
    c = csl.get("container-title") or ""
    if isinstance(c, list):
        c = c[0] if c else ""
    return c.strip()


def _is_preprint(doi: str, csl: dict) -> bool:
    if doi.lower().startswith("10.48550/arxiv"):
        return True
    if (csl.get("type") or "").lower() in ("posted-content", "article"):
        # DataCite arXiv often types as 'article'/'posted-content' w/ no journal
        return not _container(csl)
    return False


def build_apa(doi: str, csl: dict) -> dict:
    """Return {'inline', 'narrative', 'reference'} — all APA 7."""
    authors_intext = apa_authors_intext(csl)
    authors_narr = apa_authors_intext(csl, narrative=True)
    authors_ref = apa_authors_ref(csl)
    year = _year(csl)
    title = _title(csl)
    doi_url = f"https://doi.org/{doi}"

    inline = f"({authors_intext}, {year})"
    narrative = f"{authors_narr} ({year})"

    if _is_preprint(doi, csl):
        # APA 7 preprint: Authors (Year). Title [Preprint]. Repository. URL
        arxiv_id = doi.split("/")[-1].replace("arXiv.", "")
        reference = (f"{authors_ref} ({year}). {title} [Preprint]. "
                     f"arXiv. {doi_url}")
        if arxiv_id and arxiv_id.lower() != doi.lower():
            reference = (f"{authors_ref} ({year}). {title} "
                         f"(arXiv:{arxiv_id}) [Preprint]. arXiv. {doi_url}")
    else:
        container = _container(csl)
        vol = str(csl.get("volume") or "").strip()
        issue = str(csl.get("issue") or "").strip()
        pages = str(csl.get("page") or "").strip().replace("-", "–")
        loc = ""
        if vol:
            loc = f", *{vol}*"                 # italic volume
            if issue:
                loc += f"({issue})"
        elif issue:
            loc = f", ({issue})"
        if pages:
            loc += f", {pages}"
        journal = f"*{container}*" if container else ""
        reference = f"{authors_ref} ({year}). {title}."
        if journal:
            reference += f" {journal}{loc}."
        elif csl.get("publisher"):
            reference += f" {csl['publisher']}."
        reference += f" {doi_url}"

    return {
        "doi": doi,
        "inline": inline,
        "narrative": narrative,
        "reference": " ".join(reference.split()),   # collapse stray spaces
        "authors_intext": authors_intext,
        "year": year,
    }


def citation_for(name: str, cache: dict | None = None,
                 use_network: bool = True) -> dict | None:
    """One-shot: filename -> full APA citation dict, or None if unresolvable.

    On resolution failure returns a minimal stub carrying just the DOI so the
    document is still recorded (author/reference left unresolved).
    """
    doi = doi_from_filename(name)
    if not doi:
        return None
    csl = resolve_csl(doi, cache=cache, use_network=use_network)
    if csl is None:
        return {"doi": doi, "inline": f"(Unknown, n.d.)",
                "narrative": "Unknown (n.d.)", "reference": "",
                "authors_intext": "Unknown", "year": "n.d.",
                "unresolved": True}
    return build_apa(doi, csl)


if __name__ == "__main__":
    # quick manual check on the local corpus
    pdf_dir = Path(__file__).parent / "pdfs"
    shared: dict = _load_cache()
    for p in sorted(pdf_dir.glob("*.pdf")):
        c = citation_for(p.name, cache=shared)
        print(f"{p.name}\n  doi: {c and c['doi']}\n  in : {c and c['inline']}"
              f"\n  ref: {c and c['reference']}\n")
    _save_cache(shared)

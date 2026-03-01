#!/usr/bin/env python3
"""
step1_fetch_papers.py
─────────────────────
Dynamically searches Semantic Scholar for CDKL5-related papers and randomly
samples N papers with good coverage across research categories. No hardcoded
DOIs or titles — every run can produce a different set.

Strategy:
  1. Run searches across all CDKL5 research categories
  2. Pool all results, deduplicate by DOI
  3. Score each paper (open-access bonus, recency, citation count)
  4. Randomly sample N papers — one per category first, then fill remainder
     from the pool weighted by score
  5. Download PDFs for open-access papers via Semantic Scholar + Unpaywall

USAGE:
    python step1_fetch_papers.py                  # Fetch 5 papers (default)
    python step1_fetch_papers.py --count 10       # Fetch 10 papers
    python step1_fetch_papers.py --count 5 --seed 42   # Reproducible random sample
    python step1_fetch_papers.py --no-pdfs        # Metadata only, skip PDF download
    python step1_fetch_papers.py --oa-only        # Only papers with open-access PDFs
"""

import requests
import json
import time
import random
import math
import argparse
from pathlib import Path
from loguru import logger


# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path("documents")
METADATA_FILE = Path("output/parsed/papers_metadata.json")
OUTPUT_DIR.mkdir(exist_ok=True)
METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── API ───────────────────────────────────────────────────────────────────────
SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS     = "paperId,title,abstract,year,authors,openAccessPdf,externalIds,journal,citationCount,isOpenAccess"
REQUEST_DELAY = 1.2   # seconds between API calls
MIN_YEAR      = 2010  # ignore papers older than this
MIN_ABSTRACT  = 100   # ignore papers with very short/missing abstracts

# ── Research categories + queries ─────────────────────────────────────────────
# One query per category — broad enough to return many results, specific enough
# to stay on-topic. We search all categories so the final sample has diversity.
CATEGORY_QUERIES = {
    "molecular_biology":  "CDKL5 kinase signaling synaptic phosphorylation",
    "electrophysiology":  "CDKL5 electrophysiology neural recording MEA",
    "animal_models":      "CDKL5 mouse model knockout behavioral phenotype",
    "gene_therapy":       "CDKL5 gene therapy AAV antisense oligonucleotide",
    "organoids_ipsc":     "CDKL5 iPSC organoid patient-derived neurons",
    "clinical":           "CDKL5 deficiency disorder clinical natural history patient",
    "therapeutics":       "CDKL5 drug treatment pharmacological rescue",
    "behavioral":         "CDKL5 cognitive motor behavioral analysis",
}

RESULTS_PER_QUERY = 40   # Fetch this many per category before sampling


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

def search_category(category: str, query: str, limit: int = RESULTS_PER_QUERY) -> list:
    """
    Search Semantic Scholar for one category.
    Returns raw paper dicts with category tag attached.
    """
    params = {
        "query":  query,
        "fields": SS_FIELDS,
        "limit":  min(limit, 100),
    }
    try:
        response = requests.get(SS_SEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        papers = response.json().get("data", [])

        # Tag each result with its source category
        for p in papers:
            p["_category"]      = category
            p["_search_query"]  = query

        logger.info(f"   [{category}] → {len(papers)} results")
        return papers

    except requests.HTTPError as e:
        logger.warning(f"   [{category}] HTTP error: {e}")
        return []
    except Exception as e:
        logger.warning(f"   [{category}] Error: {e}")
        return []


def search_all_categories(queries: dict, limit_per: int) -> list:
    """Search every category and return the combined pool."""
    pool = []
    logger.info(f"\n🔍 Searching {len(queries)} CDKL5 research categories...\n")

    for category, query in queries.items():
        results = search_category(category, query, limit=limit_per)
        pool.extend(results)
        time.sleep(REQUEST_DELAY)

    logger.info(f"\n   Raw pool: {len(pool)} papers across {len(queries)} categories")
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Filtering + Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(papers: list) -> list:
    """
    Deduplicate by DOI. When a paper appears in multiple categories,
    keep it under the first (highest-ranked) category.
    """
    seen_dois  = set()
    seen_ids   = set()
    unique     = []

    for p in papers:
        doi      = (p.get("externalIds") or {}).get("DOI")
        paper_id = p.get("paperId")

        if doi and doi in seen_dois:
            continue
        if paper_id and paper_id in seen_ids:
            continue

        if doi:       seen_dois.add(doi)
        if paper_id:  seen_ids.add(paper_id)
        unique.append(p)

    logger.info(f"   After dedup: {len(unique)} unique papers")
    return unique


def filter_papers(papers: list, min_year: int, min_abstract: int, oa_only: bool) -> list:
    """Apply quality filters."""
    filtered = []
    for p in papers:
        year     = p.get("year") or 0
        abstract = p.get("abstract") or ""
        has_oa   = bool(p.get("openAccessPdf"))

        if year < min_year:         continue
        if len(abstract) < min_abstract: continue
        if oa_only and not has_oa:  continue
        if not p.get("title"):      continue

        filtered.append(p)

    logger.info(f"   After filters (year≥{min_year}, abstract, {'OA only' if oa_only else 'any access'}): {len(filtered)} papers")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Scoring + Sampling
# ─────────────────────────────────────────────────────────────────────────────

def score_paper(paper: dict) -> float:
    """
    Score a paper for weighted random sampling.
    Higher score = more likely to be sampled.

    Factors:
      - Open access PDF available  → large bonus (we can actually parse it)
      - More recent                → slight bonus (more relevant to current research)
      - More citations             → slight bonus (higher impact)
    """
    score = 1.0

    if paper.get("openAccessPdf"):
        score += 5.0

    year    = paper.get("year") or MIN_YEAR
    recency = max(0, year - MIN_YEAR) / max(1, 2024 - MIN_YEAR)
    score  += recency * 2.0

    citations = paper.get("citationCount") or 0
    score    += math.log1p(citations) * 0.3

    return score


def sample_with_category_coverage(papers: list, n: int, seed) -> list:
    """
    Sample N papers ensuring at least one per category if possible,
    then fill remaining slots using weighted random sampling.
    """
    rng = random.Random(seed)

    # Group by category
    by_category = {}
    for p in papers:
        cat = p.get("_category", "unknown")
        by_category.setdefault(cat, []).append(p)

    selected   = []
    used_ids   = set()
    categories = list(by_category.keys())

    # Phase 1: one weighted-random paper per category
    logger.info(f"\n🎲 Phase 1: Sampling one paper per category...")
    rng.shuffle(categories)

    for cat in categories:
        if len(selected) >= n:
            break
        candidates = [p for p in by_category[cat] if p.get("paperId") not in used_ids]
        if not candidates:
            continue

        weights = [score_paper(p) for p in candidates]
        chosen  = rng.choices(candidates, weights=weights, k=1)[0]
        selected.append(chosen)
        used_ids.add(chosen.get("paperId"))

        oa_flag = "📄 OA" if chosen.get("openAccessPdf") else "🔒 no PDF"
        logger.info(f"   [{cat}] → {chosen.get('title', '')[:60]} ({chosen.get('year')}) {oa_flag}")

    # Phase 2: fill remaining from full pool
    remaining = n - len(selected)
    if remaining > 0:
        logger.info(f"\n🎲 Phase 2: Filling {remaining} remaining slot(s) from full pool...")
        pool_rest = [p for p in papers if p.get("paperId") not in used_ids]
        weights   = [score_paper(p) for p in pool_rest]
        seen_extra = set()

        extra = rng.choices(pool_rest, weights=weights, k=min(remaining * 3, len(pool_rest)))
        for p in extra:
            pid = p.get("paperId")
            if pid not in used_ids and pid not in seen_extra:
                selected.append(p)
                seen_extra.add(pid)
                oa_flag = "📄 OA" if p.get("openAccessPdf") else "🔒 no PDF"
                logger.info(f"   [fill/{p.get('_category')}] → {p.get('title', '')[:55]} ({p.get('year')}) {oa_flag}")
            if len(selected) >= n:
                break

    return selected[:n]


# ─────────────────────────────────────────────────────────────────────────────
# PDF Download
# ─────────────────────────────────────────────────────────────────────────────

def try_unpaywall(doi: str):
    """Look up open-access PDF via Unpaywall."""
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}?email=research@cdkl5connect.org"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            best = r.json().get("best_oa_location")
            if best:
                return best.get("url_for_pdf") or best.get("url")
    except Exception:
        pass
    return None


def download_pdf(pdf_url: str, paper_id: str, title: str):
    """Download a PDF, validate it, and save to documents/."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title[:60])
    path = OUTPUT_DIR / f"{safe}.pdf"

    if path.exists():
        logger.info(f"   Already downloaded: {path.name}")
        return path

    try:
        headers  = {"User-Agent": "CDKL5-Connect Research Platform (research use)"}
        response = requests.get(pdf_url, headers=headers, timeout=45, stream=True)
        response.raise_for_status()

        with open(path, "wb") as f:
            first = True
            for chunk in response.iter_content(8192):
                if first:
                    if not chunk.startswith(b"%PDF"):
                        logger.warning(f"   Response is not a PDF for: {title[:40]}")
                        return None
                    first = False
                f.write(chunk)

        kb = path.stat().st_size // 1024
        logger.info(f"   ✅ Saved ({kb} KB): {path.name}")
        return path

    except Exception as e:
        logger.warning(f"   Download failed: {e}")
        if path.exists():
            path.unlink()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(count=5, seed=None, download_pdfs=True, oa_only=False):
    logger.info(f"🚀 CDKL5-Connect — Step 1: Dynamic paper collection (n={count})")
    if seed is not None:
        logger.info(f"   Random seed: {seed}  (reproducible run)")
    logger.info("")

    # 1. Search all categories
    raw_pool = search_all_categories(CATEGORY_QUERIES, limit_per=RESULTS_PER_QUERY)

    # 2. Deduplicate + filter
    unique   = deduplicate(raw_pool)
    filtered = filter_papers(unique, min_year=MIN_YEAR, min_abstract=MIN_ABSTRACT, oa_only=oa_only)

    if len(filtered) < count:
        logger.warning(f"   Only {len(filtered)} papers pass filters (requested {count}). Relaxing...")
        filtered = deduplicate(raw_pool)

    # 3. Sample with category coverage
    selected = sample_with_category_coverage(filtered, n=count, seed=seed)
    logger.info(f"\n   Final selection: {len(selected)} papers\n")

    # 4. Download PDFs
    collected = []

    for i, paper in enumerate(selected, 1):
        pid      = paper.get("paperId", f"unknown_{i}")
        title    = paper.get("title", "Untitled")
        doi      = (paper.get("externalIds") or {}).get("DOI")
        category = paper.get("_category", "unknown")
        authors  = [a.get("name", "") for a in (paper.get("authors") or [])]
        journal  = (paper.get("journal") or {}).get("name")

        logger.info(f"📄 [{i}/{len(selected)}] {title[:68]}")
        logger.info(f"   Category: {category} | Year: {paper.get('year')} | Citations: {paper.get('citationCount', 0)}")

        pdf_url  = None
        pdf_path = None

        if download_pdfs:
            oa = paper.get("openAccessPdf")
            if oa:
                pdf_url = oa.get("url")

            if not pdf_url and doi:
                logger.info("   No direct OA link, trying Unpaywall...")
                pdf_url = try_unpaywall(doi)
                time.sleep(0.5)

            if pdf_url:
                pdf_path = download_pdf(pdf_url, pid, title)
            else:
                logger.info("   ⚠️  No open-access PDF — will use abstract only")

        collected.append({
            "paper_id":       pid,
            "title":          title,
            "abstract":       paper.get("abstract"),
            "authors":        authors,
            "year":           paper.get("year"),
            "doi":            doi,
            "journal":        journal,
            "citation_count": paper.get("citationCount", 0),
            "open_access":    bool(paper.get("isOpenAccess")),
            "category":       category,
            "search_query":   paper.get("_search_query", ""),
            "pdf_url":        pdf_url,
            "pdf_local_path": str(pdf_path) if pdf_path else None,
            "pdf_available":  pdf_path is not None,
        })
        logger.info("")
        time.sleep(REQUEST_DELAY)

    # 5. Save metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(collected, f, indent=2)

    # Summary
    pdfs_got = sum(1 for p in collected if p["pdf_available"])
    oa_count = sum(1 for p in collected if p["open_access"])
    cats_got = set(p["category"] for p in collected)

    logger.info("─" * 58)
    logger.info("📊 Collection Summary")
    logger.info("─" * 58)
    logger.info(f"   Papers collected:    {len(collected)}")
    logger.info(f"   PDFs downloaded:     {pdfs_got}/{len(collected)}")
    logger.info(f"   Open access:         {oa_count}/{len(collected)}")
    logger.info(f"   Categories covered:  {len(cats_got)} → {', '.join(sorted(cats_got))}")
    logger.info(f"   Metadata:            {METADATA_FILE}")
    logger.info("─" * 58)
    logger.info("\n📋 Selected Papers:")
    for p in collected:
        icon = "📄" if p["pdf_available"] else "🔒"
        logger.info(f"   {icon} [{p['category']:<20}] {p['title'][:55]} ({p['year']})")
    logger.info("")
    logger.info("✅ Next step: python step2_parse_documents.py")

    return collected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDKL5-Connect — Dynamic paper fetcher")
    parser.add_argument("--count",   type=int,  default=5,    help="Number of papers to collect (default: 5)")
    parser.add_argument("--seed",    type=int,  default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-pdfs", action="store_true",     help="Skip PDF downloads (metadata only)")
    parser.add_argument("--oa-only", action="store_true",     help="Only select papers with open-access PDFs")
    args = parser.parse_args()

    main(
        count=args.count,
        seed=args.seed,
        download_pdfs=not args.no_pdfs,
        oa_only=args.oa_only,
    )
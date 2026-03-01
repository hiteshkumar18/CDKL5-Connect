#!/usr/bin/env python3
"""
step2_parse_documents.py
─────────────────────────
Uses AI2's PaperMage to parse PDFs into structured, section-aware documents.
PaperMage understands scientific paper layout — it knows the difference between
an Abstract, a Methods section, a Figure caption, and a References list.

For each PDF this produces:
  - Structured sections (Title, Abstract, Introduction, Methods, Results, etc.)
  - Figure and table captions (extracted separately)
  - Inline citations
  - Author/institution metadata
  - Clean text per section (no headers/footers/page numbers)

FALLBACK: If PaperMage isn't installed or fails on a document, falls back to
pymupdf4llm which converts PDFs to clean Markdown — still very good for RAG.

USAGE:
    python step2_parse_documents.py
"""

import json
import os
from pathlib import Path
from loguru import logger
import traceback


INPUT_METADATA  = Path("output/parsed/papers_metadata.json")
OUTPUT_DIR      = Path("output/parsed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PaperMage Parser (AI2 Primary Method)
# ─────────────────────────────────────────────────────────────────────────────

def parse_with_papermage(pdf_path: str) -> dict | None:
    """
    Parse a PDF using AI2's PaperMage.

    PaperMage returns a Document object with these key layers:
      - doc.symbols        : full text as one string
      - doc.pages          : page-level layout
      - doc.tokens         : individual word tokens with bounding boxes
      - doc.sentences      : sentence segmentation
      - doc.blocks         : text blocks (paragraphs)
      - doc.sections       : section headings + content  ← most useful for RAG
      - doc.figures        : figure regions
      - doc.tables         : table regions
      - doc.captions       : figure/table captions
      - doc.footnotes      : footnotes
      - doc.bibliographies : reference list entries
      - doc.equations      : math equations

    Each entity has:
      - .text              : clean text content
      - .boxes             : bounding boxes on the page (for visual rendering)
      - .metadata          : entity-specific metadata
    """
    try:
        from papermage import Document
        from papermage.recipes import CoreRecipe

        # CoreRecipe uses CPU-only models (no GPU required)
        # It runs: PDF parser + section detector + figure/table detector
        recipe = CoreRecipe()
        doc    = recipe.run(pdf_path)

        # ── Extract sections ────────────────────────────────────────────────
        sections = []

        if hasattr(doc, "sections") and doc.sections:
            for section in doc.sections:
                section_text = section.text.strip()
                if len(section_text) < 30:  # Skip empty or noise
                    continue
                sections.append({
                    "type":    "section",
                    "heading": getattr(section.metadata, "title", ""),
                    "text":    section_text,
                    "page":    section.boxes[0].page if section.boxes else None,
                })
        else:
            # PaperMage found no sections — fall back to raw blocks
            logger.warning("  PaperMage: no sections found, using text blocks")
            for block in doc.blocks:
                text = block.text.strip()
                if len(text) > 50:
                    sections.append({
                        "type":    "block",
                        "heading": "",
                        "text":    text,
                        "page":    block.boxes[0].page if block.boxes else None,
                    })

        # ── Extract captions (figures + tables) ────────────────────────────
        captions = []
        if hasattr(doc, "captions") and doc.captions:
            for cap in doc.captions:
                captions.append({
                    "type": "caption",
                    "text": cap.text.strip(),
                    "page": cap.boxes[0].page if cap.boxes else None,
                })

        # ── Extract bibliography ────────────────────────────────────────────
        references = []
        if hasattr(doc, "bibliographies") and doc.bibliographies:
            for ref in doc.bibliographies:
                references.append(ref.text.strip())

        # ── Full text ────────────────────────────────────────────────────────
        full_text = doc.symbols if hasattr(doc, "symbols") else ""

        return {
            "parser":     "papermage",
            "full_text":  full_text,
            "sections":   sections,
            "captions":   captions,
            "references": references,
            "n_pages":    len(doc.pages) if hasattr(doc, "pages") else None,
            "n_figures":  len(doc.figures) if hasattr(doc, "figures") else 0,
            "n_tables":   len(doc.tables) if hasattr(doc, "tables") else 0,
        }

    except ImportError:
        logger.warning("  PaperMage not installed — using pymupdf4llm fallback")
        return None
    except Exception as e:
        logger.warning(f"  PaperMage failed: {e}")
        logger.debug(traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────────────────
# pymupdf4llm Fallback (Good Markdown output)
# ─────────────────────────────────────────────────────────────────────────────

def parse_with_pymupdf4llm(pdf_path: str) -> dict | None:
    """
    Fallback parser using pymupdf4llm.
    Converts PDF to clean Markdown, preserving headings and structure.
    Very good for RAG even without AI2's layout models.
    """
    try:
        import pymupdf4llm
        import fitz  # PyMuPDF

        md_text = pymupdf4llm.to_markdown(pdf_path)

        # Parse markdown into pseudo-sections by heading
        sections = _split_markdown_sections(md_text)

        # Get page count
        doc     = fitz.open(pdf_path)
        n_pages = len(doc)
        doc.close()

        return {
            "parser":     "pymupdf4llm",
            "full_text":  md_text,
            "sections":   sections,
            "captions":   [],
            "references": [],
            "n_pages":    n_pages,
            "n_figures":  0,
            "n_tables":   0,
        }

    except ImportError:
        logger.error("  pymupdf4llm not installed either. Run: pip install pymupdf4llm")
        return None
    except Exception as e:
        logger.error(f"  pymupdf4llm also failed: {e}")
        return None


def _split_markdown_sections(markdown: str) -> list:
    """Split markdown text into sections by heading markers."""
    import re
    sections = []
    current_heading = "Introduction"
    current_lines   = []

    for line in markdown.split("\n"):
        # Detect markdown headings (# or ##)
        heading_match = re.match(r"^#{1,3}\s+(.+)$", line.strip())
        if heading_match:
            # Save previous section
            text = "\n".join(current_lines).strip()
            if len(text) > 50:
                sections.append({
                    "type":    "section",
                    "heading": current_heading,
                    "text":    text,
                    "page":    None,
                })
            current_heading = heading_match.group(1).strip()
            current_lines   = []
        else:
            current_lines.append(line)

    # Last section
    text = "\n".join(current_lines).strip()
    if len(text) > 50:
        sections.append({
            "type":    "section",
            "heading": current_heading,
            "text":    text,
            "page":    None,
        })

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("🔬 CDKL5-Connect — Step 2: Parsing documents with AI2 PaperMage\n")

    # Load metadata from step 1
    if not INPUT_METADATA.exists():
        logger.error(f"Metadata file not found: {INPUT_METADATA}")
        logger.error("Run step1_fetch_papers.py first.")
        return

    with open(INPUT_METADATA) as f:
        papers = json.load(f)

    parsed_results = []

    for i, paper in enumerate(papers, 1):
        title    = paper.get("title", "Unknown")
        pdf_path = paper.get("pdf_local_path")
        doi      = paper.get("doi")

        logger.info(f"📄 [{i}/{len(papers)}] {title[:65]}...")

        parsed = None

        if pdf_path and Path(pdf_path).exists():
            logger.info(f"   Parsing PDF: {Path(pdf_path).name}")

            # Try PaperMage first (AI2's full stack)
            parsed = parse_with_papermage(pdf_path)

            # Fallback to pymupdf4llm
            if parsed is None:
                logger.info("   Falling back to pymupdf4llm...")
                parsed = parse_with_pymupdf4llm(pdf_path)

        else:
            logger.warning("   No PDF available — using abstract only")
            # For papers without PDFs, create a minimal record from abstract
            parsed = {
                "parser":     "abstract_only",
                "full_text":  paper.get("abstract", ""),
                "sections": [
                    {
                        "type":    "section",
                        "heading": "Abstract",
                        "text":    paper.get("abstract", ""),
                        "page":    None,
                    }
                ],
                "captions":   [],
                "references": [],
                "n_pages":    0,
                "n_figures":  0,
                "n_tables":   0,
            }

        if parsed:
            # Log what was extracted
            n_sections = len(parsed.get("sections", []))
            n_pages    = parsed.get("n_pages", 0)
            n_chars    = len(parsed.get("full_text", ""))
            parser     = parsed.get("parser", "unknown")

            logger.info(f"   ✅ Parser:   {parser}")
            logger.info(f"   ✅ Sections: {n_sections} | Pages: {n_pages} | Chars: {n_chars:,}")

            # Build final output record
            result = {
                **paper,
                "parsed": parsed,
            }
            parsed_results.append(result)

            # Save individual parsed document
            safe_name  = "".join(c if c.isalnum() or c in "-_" else "_" for c in title[:60])
            out_path   = OUTPUT_DIR / f"{safe_name}_parsed.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"   💾 Saved: {out_path.name}")

        else:
            logger.error(f"   ❌ All parsers failed for: {title[:50]}")

        logger.info("")

    # Save combined output
    combined_path = OUTPUT_DIR / "all_papers_parsed.json"
    with open(combined_path, "w") as f:
        json.dump(parsed_results, f, indent=2)

    # Summary
    papermage_count = sum(1 for p in parsed_results if p.get("parsed", {}).get("parser") == "papermage")
    fallback_count  = sum(1 for p in parsed_results if p.get("parsed", {}).get("parser") == "pymupdf4llm")
    abstract_count  = sum(1 for p in parsed_results if p.get("parsed", {}).get("parser") == "abstract_only")

    logger.info(f"{'─'*55}")
    logger.info(f"📊 Parsing Summary:")
    logger.info(f"   Total parsed:    {len(parsed_results)}")
    logger.info(f"   Via PaperMage:   {papermage_count}  ← AI2 full layout parsing")
    logger.info(f"   Via pymupdf4llm: {fallback_count}  ← Markdown fallback")
    logger.info(f"   Abstract only:   {abstract_count}  ← No PDF available")
    logger.info(f"   Output:          {combined_path}")
    logger.info(f"{'─'*55}")
    logger.info(f"\n✅ Next step: python step3_chunk_documents.py")


if __name__ == "__main__":
    main()
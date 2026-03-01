#!/usr/bin/env python3
"""
parse_with_molmo.py
────────────────────
Parses scientific PDFs into Markdown using AI2's MolMo.

TWO MODES:
┌──────────────────────────────────────────────────────────────────┐
│  ONLINE  (default) — HuggingFace Inference API                   │
│    ✅ Completely free — just needs a free HF account             │
│    ✅ No GPU, no big download                                     │
│    ✅ Sign up at: https://huggingface.co/join                     │
│    ✅ Get token at: https://huggingface.co/settings/tokens        │
│    ⚠️  Rate limited on free tier (good enough for a pilot)        │
│                                                                  │
│  OFFLINE — local model, 4-bit quantized                          │
│    ✅ No API, runs forever once downloaded                        │
│    ⚠️  Needs ~4–6 GB VRAM + first-run ~28GB download             │
└──────────────────────────────────────────────────────────────────┘

SETUP:
    # Online (free):
    pip install huggingface_hub pdf2image Pillow
    export HF_TOKEN=hf_your_token_here

    # Offline (local):
    pip install transformers accelerate einops bitsandbytes pdf2image Pillow
    (no token needed)

USAGE:
    python parse_with_molmo.py                         # online, all PDFs in documents/
    python parse_with_molmo.py --pdf paper.pdf         # single PDF
    python parse_with_molmo.py --offline               # local model
    python parse_with_molmo.py --max-pages 5           # first 5 pages only
    python parse_with_molmo.py --qa "What AAV vectors are described?"

OUTPUT:
    output/molmo/
    ├── Paper_Title/
    │   ├── page_0001.md
    │   ├── page_0002.md
    │   └── full_document.md
"""

import os
import base64
import argparse
import time
import gc
from io import BytesIO
from pathlib import Path
from loguru import logger


# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR    = Path("documents")
OUTPUT_DIR = Path("output/molmo")

# HuggingFace model ID — same model, works for both online and offline
MOLMO_MODEL = "allenai/MolmoE-1B-0924"

DPI       = 150
MAX_PAGES = 50

EXTRACTION_PROMPT = """You are extracting text from a page of a scientific research paper.
Transcribe ALL text on this page into clean Markdown, preserving document structure.

- Use ## for section headings (Abstract, Introduction, Methods, Results, etc.)
- Preserve tables using Markdown table syntax (| col | col |)
- Wrap equations in $$ markers
- Write figure captions as: **Figure N:** caption text
- Output ONLY the transcribed content — no commentary

Markdown output:"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_base64(image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE — HuggingFace Inference API (free)
# ─────────────────────────────────────────────────────────────────────────────

class OnlineMolMo:
    """
    Calls MolMo via HuggingFace's free Serverless Inference API.

    Free tier limits:
      - ~1000 requests/day
      - Rate limited if too fast — the retry logic handles this automatically
      - For a 5-paper pilot (avg 15 pages each) = ~75 requests — well within limits

    Get your free token:
      1. Sign up at https://huggingface.co/join  (free, no card)
      2. Go to https://huggingface.co/settings/tokens
      3. Create a token with "Read" permission
      4. export HF_TOKEN=hf_xxxxxxxxxxxx
    """

    def __init__(self, model: str = MOLMO_MODEL):
        self.model = model
        self._setup()

    def _setup(self):
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise EnvironmentError(
                "HF_TOKEN not set.\n\n"
                "  Get a FREE token (no credit card):\n"
                "    1. Sign up:    https://huggingface.co/join\n"
                "    2. Get token:  https://huggingface.co/settings/tokens\n"
                "    3. Run:        export HF_TOKEN=hf_your_token_here\n"
            )

        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(model=self.model, token=token)
            logger.info(f"✅ HuggingFace Inference API ready")
            logger.info(f"   Model: {self.model}")
            logger.info(f"   Free tier: ~1000 requests/day\n")
        except ImportError:
            raise ImportError("Run: pip install huggingface_hub")

    def process_page(self, image, prompt: str) -> str:
        """Send a page image to MolMo via HF Inference API."""
        b64    = pil_to_base64(image)
        retries = 3

        for attempt in range(retries):
            try:
                # HF InferenceClient visual_question_answering for vision models
                result = self.client.chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type":      "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    max_tokens=2048,
                )
                return result.choices[0].message.content.strip()

            except Exception as e:
                err = str(e).lower()

                # Rate limit — wait and retry
                if "rate" in err or "429" in err or "too many" in err:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"   Rate limited — waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue

                # Model loading (HF cold-starts models)
                if "loading" in err or "503" in err:
                    wait = 20
                    logger.info(f"   Model loading on HF servers — waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # Non-retriable error
                raise

        raise RuntimeError(f"Failed after {retries} retries")


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE — Local model, 4-bit quantized
# ─────────────────────────────────────────────────────────────────────────────

class OfflineMolMo:
    """
    Runs MolMo locally using HuggingFace transformers.
    4-bit NF4 quantization cuts memory from ~28GB → ~4–6GB.
    """

    def __init__(self, model: str = MOLMO_MODEL):
        self.model_name = model
        self.model      = None
        self.processor  = None
        self.device     = None
        self._load()

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if torch.cuda.is_available():
            vram        = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.device = "cuda"
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
        else:
            self.device = "cpu"
            logger.warning("No GPU — CPU mode is slow (~3–10 min/page)")

        logger.info(f"Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        logger.info("Loading model with 4-bit quantization (~4–6 GB)...")
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=quant,
                device_map="auto",
            )
            logger.info("✅ Loaded in 4-bit NF4")
        except ImportError:
            logger.warning("bitsandbytes not installed → loading full size")
            logger.warning("For smaller memory: pip install bitsandbytes")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.model.eval()
        logger.info(f"✅ MolMo ready on: {self.device}\n")

    def process_page(self, image, prompt: str) -> str:
        import torch
        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {
            k: v.to(self.device).unsqueeze(0) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs,
                self.processor.generate_kwargs,
                max_new_tokens=2048,
            )
        generated = output[0, inputs["input_ids"].size(1):]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)

    def cleanup(self):
        import torch
        del self.model, self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Parse one PDF (shared by both modes)
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path, molmo, prompt: str, dpi: int, max_pages: int) -> Path:
    from pdf2image import convert_from_path

    out_dir = OUTPUT_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📄 {pdf_path.name}")
    logger.info(f"   Rendering at {dpi} DPI...")

    images  = convert_from_path(str(pdf_path), dpi=dpi)
    n_pages = min(len(images), max_pages)

    if len(images) > max_pages:
        logger.warning(f"   {len(images)} pages — capping at {max_pages}")
    else:
        logger.info(f"   {n_pages} page(s) to process")

    page_texts = []

    for i, image in enumerate(images[:n_pages], start=1):
        logger.info(f"   Page {i}/{n_pages}...")
        t0 = time.time()

        try:
            markdown = molmo.process_page(image, prompt)
        except Exception as e:
            logger.warning(f"   ⚠️  Page {i} failed: {e}")
            markdown = f"<!-- Page {i} — MolMo failed: {e} -->\n"

        elapsed = time.time() - t0

        page_file = out_dir / f"page_{i:04d}.md"
        page_file.write_text(markdown, encoding="utf-8")
        page_texts.append(markdown)

        logger.info(f"   ✅ {len(markdown):,} chars in {elapsed:.1f}s → {page_file.name}")

    full_doc  = "\n\n---\n\n".join(
        f"<!-- Page {i} -->\n{text}" for i, text in enumerate(page_texts, start=1)
    )
    full_path = out_dir / "full_document.md"
    full_path.write_text(full_doc, encoding="utf-8")

    logger.info(f"\n   {n_pages} page files + full_document.md saved")
    logger.info(f"   {len(full_doc):,} total chars → {out_dir}/\n")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(pdf_path=None, offline=False, model=None, dpi=DPI, max_pages=MAX_PAGES, qa_prompt=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect PDFs
    if pdf_path:
        pdfs = [Path(pdf_path)]
        if not pdfs[0].exists():
            logger.error(f"File not found: {pdf_path}")
            return
    else:
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        if not pdfs:
            logger.error(f"No PDFs in {PDF_DIR}/ — run step1_fetch_papers.py first")
            return

    model_name = model or MOLMO_MODEL
    mode       = "offline (local)" if offline else "online (HuggingFace free API)"
    prompt     = qa_prompt or EXTRACTION_PROMPT

    logger.info(f"{'─'*58}")
    logger.info(f"  MolMo PDF Parser")
    logger.info(f"  Mode:      {mode}")
    logger.info(f"  Model:     {model_name}")
    logger.info(f"  PDFs:      {len(pdfs)}  |  DPI: {dpi}  |  Max pages: {max_pages}")
    logger.info(f"{'─'*58}\n")

    # Load model
    if offline:
        try:
            molmo = OfflineMolMo(model=model_name)
        except Exception as e:
            logger.error(f"Local model failed: {e}")
            return
    else:
        try:
            molmo = OnlineMolMo(model=model_name)
        except EnvironmentError as e:
            logger.error(str(e))
            return
        except ImportError:
            logger.error("Run: pip install huggingface_hub")
            return

    # Process
    results     = []
    total_start = time.time()

    for i, pdf in enumerate(pdfs, 1):
        logger.info(f"[{i}/{len(pdfs)}] {'─'*42}")
        t0 = time.time()
        try:
            parse_pdf(pdf, molmo, prompt, dpi, max_pages)
            results.append({"pdf": pdf.name, "status": "✅", "time": time.time() - t0})
        except Exception as e:
            logger.error(f"Failed: {pdf.name} — {e}")
            results.append({"pdf": pdf.name, "status": "❌", "error": str(e)})

    if offline and hasattr(molmo, "cleanup"):
        molmo.cleanup()

    elapsed = time.time() - total_start
    logger.info(f"\n{'='*58}")
    logger.info(f"  Done in {elapsed:.0f}s  ({mode})")
    logger.info(f"{'─'*58}")
    for r in results:
        t = f"  ({r['time']:.0f}s)" if "time" in r else ""
        logger.info(f"  {r['status']}  {r['pdf']}{t}")
    logger.info(f"\n  📁 Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDFs with MolMo → Markdown")
    parser.add_argument("--pdf",       type=str,  default=None,      help="Single PDF path")
    parser.add_argument("--offline",   action="store_true",          help="Use local model instead of HF API")
    parser.add_argument("--model",     type=str,  default=None,      help="Override model (default: MolmoE-1B-0924)")
    parser.add_argument("--dpi",       type=int,  default=DPI,       help=f"Render DPI (default: {DPI})")
    parser.add_argument("--max-pages", type=int,  default=MAX_PAGES, help=f"Max pages per PDF (default: {MAX_PAGES})")
    parser.add_argument("--qa",        type=str,  default=None,      help="Ask a question about each page instead of extracting text")
    args = parser.parse_args()

    main(
        pdf_path  = args.pdf,
        offline   = args.offline,
        model     = args.model,
        dpi       = args.dpi,
        max_pages = args.max_pages,
        qa_prompt = args.qa,
    )
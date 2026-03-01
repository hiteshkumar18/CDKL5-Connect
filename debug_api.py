#!/usr/bin/env python3
"""
debug_api.py
─────────────
Tests the OpenRouter API call on a single page and prints everything —
the raw request, raw response, status code, and any error in full detail.

Run this first before parse_with_molmo.py to confirm the API is working.

USAGE:
    python debug_api.py                        # tests with a blank white image
    python debug_api.py --pdf documents/my.pdf # tests with real page 1 of a PDF
"""

import os
import base64
import json
import argparse
from io import BytesIO

# ── Check API key ─────────────────────────────────────────────────────────────
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("❌  OPENROUTER_API_KEY not set")
    print("    export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxx")
    exit(1)
print(f"✅  API key found: {api_key[:12]}...")

# ── Build a test image ────────────────────────────────────────────────────────
def get_test_image_b64(pdf_path: str = None, page: int = 1) -> str:
    if pdf_path:
        print(f"\n📄 Rendering page {page} of: {pdf_path}")
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=150, first_page=page, last_page=page)
        img = images[0]
        print(f"   Image size: {img.size}")
    else:
        print("\n🖼  Using synthetic white test image (100x100)")
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(f"   Base64 length: {len(b64)} chars")
    return b64

# ── Make the API call ─────────────────────────────────────────────────────────
def test_api(b64_image: str, model: str):
    import requests

    url     = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                    {
                        "type": "text",
                        "text": "Describe what you see in this image in one sentence.",
                    },
                ],
            }
        ],
        "max_tokens": 256,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://cdkl5-connect.research",
        "X-Title":       "CDKL5-Connect",
    }

    print(f"\n📡 Sending request to OpenRouter...")
    print(f"   Model:  {model}")
    print(f"   URL:    {url}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as e:
        print(f"\n❌  Request failed entirely: {type(e).__name__}: {e}")
        return

    # ── Print everything ──────────────────────────────────────────────────────
    print(f"\n── RAW RESPONSE ──────────────────────────────────────────")
    print(f"   Status code: {response.status_code}")
    print(f"   Headers:")
    for k, v in response.headers.items():
        print(f"     {k}: {v}")

    print(f"\n   Body:")
    try:
        body = response.json()
        print(json.dumps(body, indent=4))
    except Exception:
        print(f"   (not JSON): {response.text[:2000]}")

    print(f"\n── INTERPRETATION ────────────────────────────────────────")

    if response.status_code == 200:
        body = response.json()
        if "choices" in body and body["choices"]:
            content = body["choices"][0]["message"]["content"]
            if content and content.strip():
                print(f"✅  SUCCESS — Model responded:")
                print(f"\n   {content.strip()}\n")
            else:
                print(f"⚠️  Got 200 but content is empty or None")
                print(f"   Full choices: {body['choices']}")
        elif "error" in body:
            print(f"❌  API returned error inside 200 response:")
            print(f"   {body['error']}")
        else:
            print(f"⚠️  Unexpected 200 response shape")

    elif response.status_code == 401:
        print("❌  401 Unauthorized — API key is invalid or expired")
        print("    Check your key at: https://openrouter.ai/keys")

    elif response.status_code == 402:
        print("❌  402 Payment Required — this model needs credits")
        print("    Try a free model: --model google/gemini-2.0-flash-exp:free")

    elif response.status_code == 429:
        print("❌  429 Rate limited — wait a minute and try again")

    elif response.status_code == 503:
        print("❌  503 Model unavailable — try a different model")

    else:
        print(f"❌  Unexpected status: {response.status_code}")

    # ── Suggest alternatives if model isn't working ───────────────────────────
    if response.status_code != 200:
        print(f"\n── FREE VISION MODELS TO TRY ─────────────────────────────")
        models = [
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-flash-1.5-8b",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "qwen/qwen2-vl-7b-instruct:free",
            "mistralai/pixtral-12b:free",
        ]
        for m in models:
            print(f"   python debug_api.py --model {m}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",   type=str, default=None,                             help="PDF to test with (uses page 1)")
    parser.add_argument("--model", type=str, default="google/gemini-2.0-flash-exp:free", help="Model to test")
    args = parser.parse_args()

    b64 = get_test_image_b64(args.pdf)
    test_api(b64, args.model)
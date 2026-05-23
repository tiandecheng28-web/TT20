"""
Omni 主图生成 API — Vercel Serverless Function
POST /api/generate  接收产品图 + 文案参数 → 调用 Doubao 生图 → 返回 6 张图片
"""
import os
import base64
import httpx
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ARK_API_KEY = os.getenv("ARK_API_KEY")
API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MODEL = "doubao-seedream-5-0-lite-260128"
ROOT = Path(__file__).parent.parent


class GeneratedImage(BaseModel):
    index: int
    url: str


class GenerateResponse(BaseModel):
    success: bool
    images: list[GeneratedImage] = []
    error: Optional[str] = None


def _encode_image(file_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Encode image bytes to data URI for Doubao API."""
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


async def _call_doubao(headers: dict, payload: dict, timeout: float = 120.0) -> dict:
    """Call Doubao image generation API with retry logic."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(API_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Doubao API error: {resp.text[:200]}")
        return resp.json()


def _extract_url(result: dict) -> str:
    """Extract image URL from Doubao API response."""
    url = result.get("url")
    if url:
        return url
    data = result.get("data", [])
    if isinstance(data, list) and data:
        return data[0].get("url", "")
    raise ValueError(f"No image URL in response")


@app.get("/")
async def index():
    return FileResponse(ROOT / "index.html")


@app.get("/editor")
async def editor():
    return FileResponse(ROOT / "editor.html")


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(
    image: UploadFile = File(...),
    title: str = Form(""),
    subtitle: str = Form(""),
    selling_points: str = Form(""),
    prompt_extra: str = Form(""),
    style: str = Form("white_background"),
    count: int = Form(6),
):
    if not ARK_API_KEY:
        raise HTTPException(status_code=500, detail="ARK_API_KEY not configured")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    ref_image = _encode_image(image_bytes, image.content_type or "image/jpeg")

    # Build prompt from user inputs
    style_map = {
        "white_background": "pure white background, Amazon product photography, soft studio lighting, sharp details, no shadows on background",
        "lifestyle": "warm natural light, lifestyle home setting, editorial photography, cozy atmosphere",
        "dark_moody": "dark dramatic lighting, luxurious atmosphere, spotlight on product, premium feel",
    }
    style_prompt = style_map.get(style, style_map["white_background"])

    text_lines = []
    if title:
        text_lines.append(f"Product title: {title}")
    if subtitle:
        text_lines.append(f"Subtitle: {subtitle}")
    if selling_points:
        text_lines.append(f"Key features: {selling_points}")

    text_context = ". ".join(text_lines) if text_lines else ""

    base_prompt = (
        f"Professional product photography, Amazon main image quality. "
        f"{text_context}. "
        f"{style_prompt}. "
        f"{prompt_extra}. "
        f"High resolution, commercial composition."
    )

    api_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ARK_API_KEY}",
    }

    images: list[GeneratedImage] = []

    # Generate 6 images (API supports up to 4 per call, so 2 calls)
    batch_size = 4
    for batch_start in range(0, count, batch_size):
        batch_count = min(batch_size, count - batch_start)

        payload = {
            "model": MODEL,
            "prompt": base_prompt,
            "image": ref_image,
            "response_format": "url",
            "size": "2K",
            "watermark": False,
            "n": batch_count,
        }

        try:
            result = await _call_doubao(api_headers, payload)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Image generation failed: {str(e)}")

        # Parse results
        data_items = result.get("data", [result] if result.get("url") else [])
        for item in data_items:
            img_url = item.get("url", "")
            if img_url:
                images.append(GeneratedImage(index=len(images), url=img_url))

    if not images:
        raise HTTPException(status_code=502, detail="No images generated")

    return GenerateResponse(success=True, images=images[:count])


@app.get("/api/health")
async def health():
    key_ok = bool(ARK_API_KEY)
    key_preview = ""
    if ARK_API_KEY:
        key_preview = ARK_API_KEY[:8] + "..." + ARK_API_KEY[-4:]
    return {
        "status": "ok" if key_ok else "no_key",
        "model": MODEL,
        "key_configured": key_ok,
        "key_preview": key_preview,
        "api_url": API_URL,
    }

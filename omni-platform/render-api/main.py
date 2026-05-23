"""
Omni 生图 API — Render 免费版部署
异步模式：接收请求 → 返回任务ID → 后台生图 → 前端轮询
"""
import os
import uuid
import base64
import asyncio
import httpx
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ARK_API_KEY = os.getenv("ARK_API_KEY")
API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MODEL = "doubao-seedream-5-0-lite-260128"

# 任务存储（内存，重启消失，够用）
tasks: dict = {}


class TaskResponse(BaseModel):
    task_id: str


class TaskStatus(BaseModel):
    status: str  # pending | done | error
    images: list = []
    error: Optional[str] = None


# ── Doubao API ──────────────────────────────────────────

def _encode_image(file_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


async def _call_doubao(headers: dict, payload: dict, timeout: float = 120) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(API_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"Doubao error {resp.status_code}: {resp.text[:300]}")
        return resp.json()


async def _run_generation(task_id: str, payload: dict):
    try:
        api_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ARK_API_KEY}",
        }
        result = await _call_doubao(api_headers, payload)

        images = []
        for item in result.get("data", []):
            url = item.get("url")
            if url:
                images.append({"index": len(images), "url": url})

        tasks[task_id] = {"status": "done", "images": images, "error": None}
    except Exception as e:
        tasks[task_id] = {"status": "error", "images": [], "error": str(e)}


# ── API 端点 ─────────────────────────────────────────────

@app.post("/api/generate", response_model=TaskResponse)
async def generate(
    image: UploadFile = File(...),
    title: str = Form(""),
    subtitle: str = Form(""),
    selling_points: str = Form(""),
    prompt_extra: str = Form(""),
    style: str = Form("white_background"),
    count: int = Form(2),
):
    if not ARK_API_KEY:
        raise HTTPException(status_code=500, detail="ARK_API_KEY not configured")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    ref_image = _encode_image(image_bytes, image.content_type or "image/jpeg")

    style_map = {
        "white_background": "pure white background, Amazon product photography, soft studio lighting, sharp details, no shadows on background",
        "lifestyle": "warm natural light, lifestyle home setting, editorial photography, cozy atmosphere",
        "dark_moody": "dark dramatic lighting, luxurious atmosphere, spotlight on product, premium feel",
    }
    style_prompt = style_map.get(style, style_map["white_background"])

    text_lines = []
    if title: text_lines.append(f"Product title: {title}")
    if subtitle: text_lines.append(f"Subtitle: {subtitle}")
    if selling_points: text_lines.append(f"Key features: {selling_points}")
    text_context = ". ".join(text_lines) if text_lines else ""

    base_prompt = (
        f"Professional product photography, Amazon main image quality. "
        f"{text_context}. "
        f"{style_prompt}. "
        f"{prompt_extra}. "
        f"High resolution, commercial photography, clean composition."
    )

    doubao_payload = {
        "model": MODEL,
        "prompt": base_prompt,
        "image": ref_image,
        "response_format": "url",
        "size": "2K",
        "watermark": False,
        "n": count,
    }

    # 异步后台生图
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {"status": "pending", "images": [], "error": None}
    asyncio.create_task(_run_generation(task_id, doubao_payload))

    return TaskResponse(task_id=task_id)


@app.get("/api/status/{task_id}", response_model=TaskStatus)
async def status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return TaskStatus(status="not_found")
    return TaskStatus(status=task["status"], images=task.get("images", []), error=task.get("error"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL, "key_configured": bool(ARK_API_KEY)}

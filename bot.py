"""
Tamatar-Laal Manga Translator Bot
==================================
Telegram bot that lets users upload a manga (image / document / zip / cbz / pdf),
configure the full manga-image-translator settings via inline menus, then
dispatches a GitHub Actions workflow (worker.py) to do the actual OCR + translate + render.

Commands:
  /start        - intro
  /settings     - open the settings menu (detector / ocr / translator / inpainter / colorizer / render / upscale / misc)
  /translate    - start a translate job (attach file in same message, or reply to a file, or send file next)
  /cancel       - cancel a pending/running job
  /myjobs       - list your active jobs
  /status <id>  - JSON status of a job
"""

import os
import json
import time
import uuid
import base64
import asyncio
import logging
from typing import Optional

import requests
from pyrogram import Client, filters
from pyrogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
)
import pyrogram.utils

pyrogram.utils.get_peer_type = lambda p: (
    "channel" if str(p).startswith("-100") else "chat" if str(p).startswith("-") else "user"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tamatar-bot")

# ============================================================================
# CONFIG (env vars only — no secrets hardcoded)
# ============================================================================
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "").strip()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
REPO_NAME = os.getenv("REPO_NAME", "").strip()          # e.g. "youruser/yourrepo"
WORKFLOW_FILE = os.getenv("WORKFLOW_FILE", "manga.yml").strip()
WORKFLOW_REF = os.getenv("WORKFLOW_REF", "main").strip()

# Only the owner/admin user id(s) may use privileged commands. Comma-separated.
OWNER_IDS = {int(x) for x in os.getenv("OWNER_IDS", "").replace(" ", "").split(",") if x}

# Where job state is kept (per-process; for multi-instance use a DB instead)
JOBS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs.json")
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_settings.json")
FONTS_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts_registry.json")
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

if not (API_ID and API_HASH and BOT_TOKEN and REPO_NAME):
    raise SystemExit(
        "Missing required env vars. Need API_ID, API_HASH, BOT_TOKEN, REPO_NAME "
        "(and GITHUB_TOKEN to actually dispatch workflows)."
    )

bot = Client("TamatarLaalBot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# ============================================================================
# FULL manga-image-translator SETTINGS SCHEMA
# (mirrors zyddnys/manga-image-translator config-help JSON schema)
# ============================================================================

DETECTORS = ["default", "dbconvnext", "ctd", "craft", "paddle", "none"]
OCR_MODELS = ["32px", "48px", "48px_ctc", "mocr"]
INPAINTERS = ["default", "lama_large", "lama_mpe", "sd", "none", "original"]
COLORIZERS = ["none", "mc2"]
UPSCALERS = ["waifu2x", "esrgan", "4xultrasharp"]
RENDERERS = ["default", "manga2eng", "none"]
ALIGNMENTS = ["auto", "left", "center", "right"]
DIRECTIONS = ["auto", "horizontal", "vertical"]
INPAINT_PRECISIONS = ["fp32", "fp16", "bf16"]

TRANSLATORS = [
    "youdao", "baidu", "deepl", "papago", "caiyun", "chatgpt", "none", "original",
    "sakura", "deepseek", "groq", "custom_openai", "offline", "nllb", "nllb_big",
    "sugoi", "jparacrawl", "jparacrawl_big", "m2m100", "m2m100_big", "mbart50",
    "qwen2", "qwen2_big",
]

LANGUAGES = {
    "CHS": "Simplified Chinese", "CHT": "Traditional Chinese", "CSY": "Czech",
    "NLD": "Dutch", "ENG": "English", "FRA": "French", "DEU": "German",
    "HUN": "Hungarian", "ITA": "Italian", "JPN": "Japanese", "KOR": "Korean",
    "POL": "Polish", "PTB": "Portuguese (BR)", "ROM": "Romanian", "RUS": "Russian",
    "ESP": "Spanish", "TRK": "Turkish", "UKR": "Ukrainian", "VIN": "Vietnamese",
    "ARA": "Arabic", "SRP": "Serbian", "HRV": "Croatian", "THA": "Thai",
    "IND": "Indonesian", "FIL": "Filipino",
}

DEFAULT_SETTINGS = {
    "detector": "default",
    "detection_size": 2048,
    "text_threshold": 0.5,
    "det_rotate": False,
    "det_auto_rotate": False,
    "det_invert": False,
    "det_gamma_correct": False,
    "box_threshold": 0.75,
    "unclip_ratio": 2.3,

    "ocr": "48px",
    "use_mocr_merge": False,
    "min_text_length": 0,
    "ignore_bubble": 0,

    "inpainter": "lama_large",
    "inpainting_size": 2048,
    "inpainting_precision": "bf16",

    "colorizer": "none",
    "colorization_size": 576,
    "denoise_sigma": 30,

    "upscaler": "esrgan",
    "upscale_ratio": None,
    "revert_upscaling": False,

    "translator": "sugoi",
    "target_lang": "ENG",
    "no_text_lang_skip": False,

    "renderer": "default",
    "alignment": "auto",
    "direction": "auto",
    "disable_font_border": False,
    "font_size_offset": 0,
    "font_size_minimum": -1,
    "uppercase": False,
    "lowercase": False,
    "no_hyphenation": False,
    "rtl": True,
    "font_path": None,   # None = original/default engine font. Else path to a user-uploaded .ttf/.otf
    "font_name": None,   # display name shown in menus for the selected font

    # custom_openai translator (OpenAI-compatible endpoint)
    "custom_openai_api_base": None,   # e.g. https://api.groq.com/openai/v1 or http://localhost:11434/v1
    "custom_openai_model": None,      # e.g. gpt-4o-mini, qwen2.5:7b, etc.
    "custom_openai_api_key": None,    # optional depending on provider

    "kernel_size": 3,
    "mask_dilation_offset": 30,

    # bot-level (not library flags)
    "human_translation_mode": False,   # optional: route through PM 10-min manual translation flow
}

# ============================================================================
# PERSISTENCE HELPERS
# ============================================================================

def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def get_user_settings(user_id: int) -> dict:
    all_settings = _load_json(SETTINGS_PATH, {})
    s = dict(DEFAULT_SETTINGS)
    s.update(all_settings.get(str(user_id), {}))
    return s


def set_user_setting(user_id: int, key: str, value):
    all_settings = _load_json(SETTINGS_PATH, {})
    u = all_settings.get(str(user_id), {})
    u[key] = value
    all_settings[str(user_id)] = u
    _save_json(SETTINGS_PATH, all_settings)


def reset_user_settings(user_id: int):
    all_settings = _load_json(SETTINGS_PATH, {})
    all_settings.pop(str(user_id), None)
    _save_json(SETTINGS_PATH, all_settings)


# ============================================================================
# FONT LIBRARY (per-user uploaded .ttf/.otf files)
# ============================================================================

def get_user_fonts(user_id: int) -> list:
    """Returns list of {id, name, filename, path} dicts for this user's uploaded fonts."""
    reg = _load_json(FONTS_REGISTRY_PATH, {})
    return reg.get(str(user_id), [])


def add_user_font(user_id: int, display_name: str, filename: str, path: str, repo_path: Optional[str] = None) -> dict:
    reg = _load_json(FONTS_REGISTRY_PATH, {})
    fonts = reg.get(str(user_id), [])
    font_id = uuid.uuid4().hex[:8]
    entry = {"id": font_id, "name": display_name, "filename": filename, "path": path, "repo_path": repo_path}
    fonts.append(entry)
    reg[str(user_id)] = fonts
    _save_json(FONTS_REGISTRY_PATH, reg)
    return entry


def get_user_font(user_id: int, font_id: str) -> Optional[dict]:
    for f in get_user_fonts(user_id):
        if f["id"] == font_id:
            return f
    return None


def push_font_to_repo(user_id: int, filename: str, local_path: str) -> Optional[str]:
    """
    Uploads the font file into the bot's repo (GitHub Contents API) so the
    GitHub Actions runner (a completely separate machine) can access it.
    Returns the in-repo path to pass as --font-path, or None on failure.
    """
    if not GITHUB_TOKEN or not REPO_NAME:
        log.error("GITHUB_TOKEN/REPO_NAME not set — cannot push font to repo")
        return None

    repo_path = f"user_fonts/{user_id}/{filename}"
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{repo_path}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    try:
        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {"message": f"Add font {filename} for user {user_id}", "content": content_b64}
        res = requests.put(url, headers=headers, json=payload, timeout=20)
        if res.status_code in (200, 201):
            return repo_path
        log.error("Font push failed: %s %s", res.status_code, res.text[:300])
        return None
    except Exception as e:
        log.error("Font push exception: %s", e)
        return None


def delete_font_from_repo(user_id: int, filename: str) -> bool:
    if not GITHUB_TOKEN or not REPO_NAME:
        return False
    repo_path = f"user_fonts/{user_id}/{filename}"
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{repo_path}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return False
        sha = res.json().get("sha")
        dres = requests.delete(url, headers=headers, json={"message": f"Remove font {filename}", "sha": sha}, timeout=10)
        return dres.status_code in (200, 204)
    except Exception:
        return False


def delete_user_font(user_id: int, font_id: str) -> bool:
    reg = _load_json(FONTS_REGISTRY_PATH, {})
    fonts = reg.get(str(user_id), [])
    target = next((f for f in fonts if f["id"] == font_id), None)
    if not target:
        return False
    fonts = [f for f in fonts if f["id"] != font_id]
    reg[str(user_id)] = fonts
    _save_json(FONTS_REGISTRY_PATH, reg)
    try:
        if os.path.exists(target["path"]):
            os.remove(target["path"])
    except Exception:
        pass
    if target.get("repo_path"):
        delete_font_from_repo(user_id, target["filename"])
    # If this font was the active selection, revert to original/default
    s = get_user_settings(user_id)
    if s.get("font_path") == target.get("repo_path"):
        set_user_setting(user_id, "font_path", None)
        set_user_setting(user_id, "font_name", None)
    return True


def load_jobs() -> dict:
    return _load_json(JOBS_PATH, {})


def save_job(job_id: str, data: dict):
    jobs = load_jobs()
    jobs[job_id] = data
    _save_json(JOBS_PATH, jobs)


def get_job(job_id: str) -> Optional[dict]:
    return load_jobs().get(job_id)


def update_job(job_id: str, **kwargs):
    jobs = load_jobs()
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        jobs[job_id]["updated_at"] = int(time.time())
        _save_json(JOBS_PATH, jobs)


# ============================================================================
# GITHUB WORKFLOW DISPATCH (fire-and-forget — no waiting here)
# ============================================================================

def dispatch_github_workflow(job_id: str, file_id: str, chat_id: int, msg_id: int,
                              user_id: int, fname: str, settings: dict) -> bool:
    """
    Fires the GitHub Actions workflow_dispatch event. Does NOT wait for it to finish.
    worker.py picks up the inputs and does the actual OCR/translate/render, then
    posts results back to Telegram itself (as it already does).
    """
    if not GITHUB_TOKEN:
        log.error("GITHUB_TOKEN not set — cannot dispatch workflow")
        return False

    url = f"https://api.github.com/repos/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    payload = {
        "ref": WORKFLOW_REF,
        "inputs": {
            "job_id": job_id,
            "file_id": file_id,
            "chat_id": str(chat_id),
            "msg_id": str(msg_id),
            "user_id": str(user_id),
            "fname": fname,
            "config_json": json.dumps(settings, ensure_ascii=False),
        },
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        return res.status_code in (200, 201, 204)
    except Exception as e:
        log.error("Failed to dispatch workflow: %s", e)
        return False


def cancel_github_workflow_run(job_id: str) -> bool:
    """
    Best-effort cancel: looks up the most recent workflow run matching this job_id
    (via the run name / job_id input) and cancels it.
    """
    if not GITHUB_TOKEN:
        return False
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    try:
        runs_url = f"https://api.github.com/repos/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/runs?per_page=20"
        res = requests.get(runs_url, headers=headers, timeout=15)
        if res.status_code != 200:
            return False
        for run in res.json().get("workflow_runs", []):
            # worker.py / workflow should echo job_id into run name for this to match
            if job_id in (run.get("name") or "") or job_id in (run.get("display_title") or ""):
                cancel_url = run["url"] + "/cancel"
                cres = requests.post(cancel_url, headers=headers, timeout=15)
                return cres.status_code in (200, 202)
        return False
    except Exception as e:
        log.error("Failed to cancel workflow: %s", e)
        return False


# ============================================================================
# SETTINGS MENU (inline keyboards)
# ============================================================================

def main_settings_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔍  Detector", callback_data="menu:detector"),
         InlineKeyboardButton("🔤  OCR", callback_data="menu:ocr")],
        [InlineKeyboardButton("🌐  Translator", callback_data="menu:translator"),
         InlineKeyboardButton("🎨  Inpainter", callback_data="menu:inpainter")],
        [InlineKeyboardButton("🖌  Colorizer", callback_data="menu:colorizer"),
         InlineKeyboardButton("📐  Upscaler", callback_data="menu:upscaler")],
        [InlineKeyboardButton("✍️  Render & Font", callback_data="menu:render"),
         InlineKeyboardButton("⚙️  Misc", callback_data="menu:misc")],
        [InlineKeyboardButton("👤  Human-TL Mode", callback_data="menu:humantl")],
        [InlineKeyboardButton("📋  View Current Config", callback_data="menu:view")],
        [InlineKeyboardButton("♻️  Reset to Defaults", callback_data="menu:reset")],
        [InlineKeyboardButton("✅  Close", callback_data="menu:close")],
    ])


def enum_kb(options, current, prefix, back="menu:main", per_row=2):
    rows = []
    row = []
    for opt in options:
        label = f"● {opt}" if opt == current else opt
        row.append(InlineKeyboardButton(label, callback_data=f"{prefix}:{opt}"))
        if len(row) == per_row:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️  Back", callback_data=back)])
    return InlineKeyboardMarkup(rows)


def bool_kb(key, current, back="menu:misc"):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(("✅ On" if current else "On"), callback_data=f"set:{key}:true"),
         InlineKeyboardButton(("✅ Off" if not current else "Off"), callback_data=f"set:{key}:false")],
        [InlineKeyboardButton("⬅️  Back", callback_data=back)],
    ])


def lang_kb(current):
    rows = []
    row = []
    for code, name in LANGUAGES.items():
        label = f"✅ {code}" if code == current else code
        row.append(InlineKeyboardButton(label, callback_data=f"set:target_lang:{code}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu:translator")])
    return InlineKeyboardMarkup(rows)


def misc_menu_kb(s):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🧩  Kernel Size  ·  {s['kernel_size']}", callback_data="numeric:kernel_size")],
        [InlineKeyboardButton(f"🎭  Mask Dilation  ·  {s['mask_dilation_offset']}", callback_data="numeric:mask_dilation_offset")],
        [InlineKeyboardButton("⬅️  Back", callback_data="menu:main")],
    ])


def detector_menu_kb(s):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🧭  Model  ·  {s['detector']}", callback_data="menu:detector_pick")],
        [InlineKeyboardButton(f"📏  Detection Size  ·  {s['detection_size']}", callback_data="numeric:detection_size")],
        [InlineKeyboardButton(f"🎚  Text Threshold  ·  {s['text_threshold']}", callback_data="numeric:text_threshold")],
        [InlineKeyboardButton(f"📦  Box Threshold  ·  {s['box_threshold']}", callback_data="numeric:box_threshold")],
        [InlineKeyboardButton(f"🔗  Unclip Ratio  ·  {s['unclip_ratio']}", callback_data="numeric:unclip_ratio")],
        [InlineKeyboardButton(f"{'✅' if s['det_rotate'] else '⬜'}  Rotate", callback_data="toggle:det_rotate"),
         InlineKeyboardButton(f"{'✅' if s['det_auto_rotate'] else '⬜'}  Auto-Rotate", callback_data="toggle:det_auto_rotate")],
        [InlineKeyboardButton(f"{'✅' if s['det_invert'] else '⬜'}  Invert", callback_data="toggle:det_invert"),
         InlineKeyboardButton(f"{'✅' if s['det_gamma_correct'] else '⬜'}  Gamma Correct", callback_data="toggle:det_gamma_correct")],
        [InlineKeyboardButton("⬅️  Back", callback_data="menu:main")],
    ])


def render_menu_kb(s):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🖋  Renderer  ·  {s['renderer']}", callback_data="menu:renderer_pick")],
        [InlineKeyboardButton(f"↔️  Alignment  ·  {s['alignment']}", callback_data="menu:alignment_pick")],
        [InlineKeyboardButton(f"🧭  Direction  ·  {s['direction']}", callback_data="menu:direction_pick")],
        [InlineKeyboardButton(f"{'✅' if s['uppercase'] else '⬜'}  Uppercase", callback_data="toggle:uppercase"),
         InlineKeyboardButton(f"{'✅' if s['lowercase'] else '⬜'}  Lowercase", callback_data="toggle:lowercase")],
        [InlineKeyboardButton(f"{'✅' if s['disable_font_border'] else '⬜'}  Disable Font Border", callback_data="toggle:disable_font_border")],
        [InlineKeyboardButton(f"{'✅' if s['no_hyphenation'] else '⬜'}  No Hyphenation", callback_data="toggle:no_hyphenation"),
         InlineKeyboardButton(f"{'✅' if s['rtl'] else '⬜'}  RTL", callback_data="toggle:rtl")],
        [InlineKeyboardButton(f"🔠  Font Size Offset  ·  {s['font_size_offset']}", callback_data="numeric:font_size_offset")],
        [InlineKeyboardButton(f"🔡  Font Size Minimum  ·  {s['font_size_minimum']}", callback_data="numeric:font_size_minimum")],
        [InlineKeyboardButton(f"🔤  Font  ·  {s['font_name'] or 'Original (Default)'}", callback_data="menu:font")],
        [InlineKeyboardButton("⬅️  Back", callback_data="menu:main")],
    ])


def inpainter_menu_kb(s):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎨  Model  ·  {s['inpainter']}", callback_data="menu:inpainter_pick")],
        [InlineKeyboardButton(f"🎯  Precision  ·  {s['inpainting_precision']}", callback_data="menu:precision_pick")],
        [InlineKeyboardButton(f"📏  Inpainting Size  ·  {s['inpainting_size']}", callback_data="numeric:inpainting_size")],
        [InlineKeyboardButton("⬅️  Back", callback_data="menu:main")],
    ])


def colorizer_menu_kb(s):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🖌  Model  ·  {s['colorizer']}", callback_data="menu:colorizer_pick")],
        [InlineKeyboardButton(f"📏  Colorization Size  ·  {s['colorization_size']}", callback_data="numeric:colorization_size")],
        [InlineKeyboardButton(f"🌫  Denoise Sigma  ·  {s['denoise_sigma']}", callback_data="numeric:denoise_sigma")],
        [InlineKeyboardButton("⬅️  Back", callback_data="menu:main")],
    ])


def upscaler_menu_kb(s):
    ratio = s["upscale_ratio"] or "None"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"📐  Model  ·  {s['upscaler']}", callback_data="menu:upscaler_pick")],
        [InlineKeyboardButton(f"🔍  Upscale Ratio  ·  {ratio}", callback_data="numeric:upscale_ratio")],
        [InlineKeyboardButton(f"{'✅' if s['revert_upscaling'] else '⬜'}  Revert After", callback_data="toggle:revert_upscaling")],
        [InlineKeyboardButton("⬅️  Back", callback_data="menu:main")],
    ])


def font_menu_kb(s, fonts):
    rows = []
    is_original = not s.get("font_path")
    rows.append([InlineKeyboardButton(
        ("● " if is_original else "○ ") + "Original (Default)",
        callback_data="set:font_path:__default__",
    )])
    for f in fonts:
        selected = f.get("repo_path") and s.get("font_path") == f["repo_path"]
        label = ("● " if selected else "○ ") + f["name"] + ("" if f.get("repo_path") else "  ⚠️")
        rows.append([
            InlineKeyboardButton(label, callback_data=f"selfont:{f['id']}"),
            InlineKeyboardButton("🗑", callback_data=f"delfont:{f['id']}"),
        ])
    rows.append([InlineKeyboardButton("➕  Add Font  (send .ttf/.otf)", callback_data="menu:font_add")])
    rows.append([InlineKeyboardButton("⬅️  Back", callback_data="menu:render")])
    return InlineKeyboardMarkup(rows)


def ocr_menu_kb(s):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🔤  Model  ·  {s['ocr']}", callback_data="menu:ocr_pick")],
        [InlineKeyboardButton(f"{'✅' if s['use_mocr_merge'] else '⬜'}  MOCR Merge", callback_data="toggle:use_mocr_merge")],
        [InlineKeyboardButton(f"📏  Min Text Length  ·  {s['min_text_length']}", callback_data="numeric:min_text_length")],
        [InlineKeyboardButton(f"🫧  Ignore Bubble  ·  {s['ignore_bubble']}", callback_data="numeric:ignore_bubble")],
        [InlineKeyboardButton("⬅️  Back", callback_data="menu:main")],
    ])


def translator_menu_kb(s):
    rows = [
        [InlineKeyboardButton(f"🌐  Engine  ·  {s['translator']}", callback_data="menu:translator_pick")],
        [InlineKeyboardButton(f"🎯  Target Language  ·  {s['target_lang']}", callback_data="menu:lang_pick")],
        [InlineKeyboardButton(f"{'✅' if s['no_text_lang_skip'] else '⬜'}  No Text-Lang Skip", callback_data="toggle:no_text_lang_skip")],
    ]
    if s["translator"] == "custom_openai":
        rows.append([InlineKeyboardButton("🔑  Custom API Settings", callback_data="menu:custom_api")])
    rows.append([InlineKeyboardButton("⬅️  Back", callback_data="menu:main")])
    return InlineKeyboardMarkup(rows)


def custom_api_menu_kb(s):
    base = s.get("custom_openai_api_base") or "not set"
    model = s.get("custom_openai_model") or "not set"
    key = "•••• set" if s.get("custom_openai_api_key") else "not set"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🌐  Base URL  ·  {base}", callback_data="text:custom_openai_api_base")],
        [InlineKeyboardButton(f"🧠  Model  ·  {model}", callback_data="text:custom_openai_model")],
        [InlineKeyboardButton(f"🔑  API Key  ·  {key}", callback_data="text:custom_openai_api_key")],
        [InlineKeyboardButton("⬅️  Back", callback_data="menu:translator")],
    ])


def humantl_kb(s):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"{'✅' if s['human_translation_mode'] else '⬜'}  Human-in-loop Translation",
            callback_data="toggle:human_translation_mode")],
        [InlineKeyboardButton(
            "ℹ️  When ON: bot sends extracted dialogue file to your PM, "
            "waits up to 10 min for you to send back translations.",
            callback_data="noop")],
        [InlineKeyboardButton("⬅️  Back", callback_data="menu:main")],
    ])


# Track which numeric field a user is currently being asked to type a value for
PENDING_NUMERIC_INPUT: dict[int, str] = {}
# Track which free-text field (e.g. custom API base/model/key) a user is typing
PENDING_TEXT_INPUT: dict[int, str] = {}
# Track pending file uploads waiting to start a job (user_id -> True once /translate called)
PENDING_TRANSLATE: dict[int, bool] = {}
# Track pending font uploads (user_id -> True once "Add Font" tapped)
PENDING_FONT_UPLOAD: dict[int, bool] = {}


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

@bot.on_message(filters.command("start"))
async def cmd_start(_, message: Message):
    await message.reply_text(
        "👋 **Tamatar-Laal Manga Translator Bot**\n\n"
        "Send me a manga page (image), a document (zip/cbz/pdf), or use /translate "
        "and then send the file.\n\n"
        "Use /settings to configure OCR, detector, translator, inpainter, colorizer, "
        "upscaler, render and custom font options before translating.\n\n"
        "**Commands:**\n"
        "/translate — start a translate job\n"
        "/settings — configure translation options\n"
        "/myjobs — list your active jobs\n"
        "/status <job_id> — get JSON status of a job\n"
        "/cancel <job_id> — cancel a running job\n",
    )


@bot.on_message(filters.command("settings"))
async def cmd_settings(_, message: Message):
    await message.reply_text(
        "⚙️ **Settings Menu**\nConfigure any part of the translation pipeline below.",
        reply_markup=main_settings_kb(),
    )


@bot.on_message(filters.command("myjobs"))
async def cmd_myjobs(_, message: Message):
    user_id = message.from_user.id
    jobs = load_jobs()
    mine = {jid: j for jid, j in jobs.items() if j.get("user_id") == user_id
            and j.get("status") not in ("done", "failed", "cancelled")}
    if not mine:
        await message.reply_text("You have no active jobs.")
        return
    lines = ["📋 **Your active jobs:**\n"]
    for jid, j in mine.items():
        lines.append(f"`{jid}` — {j.get('status', 'unknown')} — {j.get('fname', '?')}")
    await message.reply_text("\n".join(lines))


@bot.on_message(filters.command("status"))
async def cmd_status(_, message: Message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.reply_text("Usage: `/status <job_id>`")
        return
    job_id = parts[1].strip()
    job = get_job(job_id)
    if not job:
        await message.reply_text(json.dumps({"error": "job not found", "job_id": job_id}, indent=2))
        return
    await message.reply_text(f"```json\n{json.dumps(job, indent=2, ensure_ascii=False)}\n```")


@bot.on_message(filters.command("cancel"))
async def cmd_cancel(_, message: Message):
    user_id = message.from_user.id
    parts = message.text.split(maxsplit=1)
    jobs = load_jobs()

    if len(parts) >= 2:
        job_id = parts[1].strip()
        job = jobs.get(job_id)
        if not job:
            await message.reply_text(json.dumps({"error": "job not found", "job_id": job_id}))
            return
        if job.get("user_id") != user_id and user_id not in OWNER_IDS:
            await message.reply_text("❌ You can only cancel your own jobs.")
            return
    else:
        # Cancel the user's most recent active job
        mine = [(jid, j) for jid, j in jobs.items()
                if j.get("user_id") == user_id and j.get("status") not in ("done", "failed", "cancelled")]
        if not mine:
            PENDING_TRANSLATE.pop(user_id, None)
            PENDING_NUMERIC_INPUT.pop(user_id, None)
            await message.reply_text("No active job to cancel. Cleared any pending translate request.")
            return
        job_id, job = sorted(mine, key=lambda x: x[1].get("created_at", 0))[-1]

    ok = cancel_github_workflow_run(job_id)
    update_job(job_id, status="cancelled")
    await message.reply_text(
        json.dumps({"job_id": job_id, "cancelled": True, "workflow_cancel_requested": ok}, indent=2)
    )


@bot.on_message(filters.command("translate"))
async def cmd_translate(_, message: Message):
    user_id = message.from_user.id

    # If a file is attached to the same command message, start immediately
    if message.document or message.photo or message.video:
        await start_translate_job(message)
        return

    # If replying to a message with a file
    if message.reply_to_message and (
        message.reply_to_message.document or message.reply_to_message.photo
    ):
        await start_translate_job(message.reply_to_message)
        return

    PENDING_TRANSLATE[user_id] = True
    await message.reply_text(
        "📤 Send me the manga file now — image, document, zip, cbz, or pdf.\n"
        "No timers, no waiting: I'll start as soon as you send it.\n"
        "Use /cancel anytime to abort."
    )


# ============================================================================
# FILE INTAKE (image / document / video-as-doc)
# ============================================================================

FONT_EXTENSIONS = (".ttf", ".otf")


@bot.on_message(filters.private & filters.document & filters.create(
    lambda _, __, m: (m.document.file_name or "").lower().endswith(FONT_EXTENSIONS)
))
async def on_font_upload(_, message: Message):
    user_id = message.from_user.id
    if not PENDING_FONT_UPLOAD.get(user_id):
        # A .ttf/.otf was sent but user never tapped "Add Font" — don't
        # silently swallow it as a translate job either, just guide them.
        await message.reply_text(
            "That looks like a font file. Open /settings → ✍️ Render → 🔤 Font → "
            "➕ Add Font if you want to add it to your font library."
        )
        return
    PENDING_FONT_UPLOAD.pop(user_id, None)

    fname = message.document.file_name
    ext = os.path.splitext(fname)[1].lower()
    safe_name = f"{user_id}_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = os.path.join(FONTS_DIR, safe_name)

    try:
        await message.download(file_name=dest_path)
    except Exception as e:
        await message.reply_text(f"❌ Failed to save font: {e}")
        return

    status = await message.reply_text("⬆️ Uploading font to repo (needed for the worker to access it)...")
    repo_path = push_font_to_repo(user_id, safe_name, dest_path)
    if not repo_path:
        await status.edit_text(
            "❌ Saved locally but failed to push to GitHub repo — the translate worker "
            "runs on GitHub Actions and won't be able to see this font until push succeeds. "
            "Check GITHUB_TOKEN / REPO_NAME and try again."
        )
        return

    entry = add_user_font(user_id, display_name=fname, filename=safe_name, path=dest_path, repo_path=repo_path)
    # Auto-select the newly added font (worker uses the in-repo path)
    set_user_setting(user_id, "font_path", entry["repo_path"])
    set_user_setting(user_id, "font_name", entry["name"])

    await status.edit_text(
        f"✅ Font added and selected: `{entry['name']}`\n"
        f"Open /settings → ✍️ Render → 🔤 Font to manage your fonts."
    )


@bot.on_message(filters.private & (filters.document | filters.photo))
async def on_file_received(_, message: Message):
    user_id = message.from_user.id
    if not PENDING_TRANSLATE.get(user_id):
        # Not explicitly requested via /translate — ignore silently to avoid
        # accidentally starting jobs on random forwarded files.
        # Still let them know how to start one.
        await message.reply_text(
            "Got your file — send /translate (or reply to this file with /translate) to start."
        )
        return
    PENDING_TRANSLATE.pop(user_id, None)
    await start_translate_job(message)


async def start_translate_job(message: Message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    if message.document:
        file_id = message.document.file_id
        fname = message.document.file_name or f"manga_{int(time.time())}.zip"
    elif message.photo:
        file_id = message.photo.file_id
        fname = f"page_{int(time.time())}.jpg"
    elif message.video:
        file_id = message.video.file_id
        fname = message.video.file_name or f"video_{int(time.time())}.mp4"
    else:
        await message.reply_text(json.dumps({"error": "unsupported file type"}))
        return

    job_id = uuid.uuid4().hex[:12]
    settings = get_user_settings(user_id)

    status_msg = await message.reply_text(
        f"⚡ **Job Queued**\n`[░░░░░░░░░░] 0%`\nJob ID: `{job_id}`"
    )

    job_record = {
        "job_id": job_id,
        "user_id": user_id,
        "chat_id": chat_id,
        "status_msg_id": status_msg.id,
        "fname": fname,
        "file_id": file_id,
        "settings": settings,
        "status": "dispatched",
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    save_job(job_id, job_record)

    ok = dispatch_github_workflow(
        job_id=job_id,
        file_id=file_id,
        chat_id=chat_id,
        msg_id=status_msg.id,
        user_id=user_id,
        fname=fname,
        settings=settings,
    )

    if not ok:
        update_job(job_id, status="failed", error="workflow dispatch failed")
        await status_msg.edit_text(
            f"❌ **Failed to start job**\n"
            f"```json\n{json.dumps({'job_id': job_id, 'error': 'GitHub workflow dispatch failed — check GITHUB_TOKEN / REPO_NAME'}, indent=2)}\n```"
        )
        return

    await status_msg.edit_text(
        f"✅ **Job dispatched to worker**\n"
        f"Job ID: `{job_id}`\n"
        f"File: `{fname}`\n\n"
        f"I won't wait around blocking you — the worker will edit this message with live progress, "
        f"and post the final result (+ JSON) here when done.\n"
        f"Use `/status {job_id}` anytime, or /cancel to stop it."
    )


# ============================================================================
# CALLBACK QUERY HANDLER (settings menu navigation)
# ============================================================================

@bot.on_callback_query()
async def on_callback(_, cq: CallbackQuery):
    data = cq.data
    user_id = cq.from_user.id
    s = get_user_settings(user_id)

    try:
        if data == "noop":
            await cq.answer()
            return

        if data == "menu:main":
            await cq.message.edit_text("⚙️ **Settings Menu**", reply_markup=main_settings_kb())

        elif data == "menu:close":
            await cq.message.edit_text("Settings closed. Use /settings to reopen.")

        elif data == "menu:view":
            await cq.message.edit_text(
                f"```json\n{json.dumps(s, indent=2, ensure_ascii=False)}\n```",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="menu:main")]]),
            )

        elif data == "menu:reset":
            reset_user_settings(user_id)
            await cq.answer("Settings reset to defaults", show_alert=True)
            await cq.message.edit_text("⚙️ **Settings Menu**", reply_markup=main_settings_kb())

        elif data == "menu:detector":
            await cq.message.edit_text("🔍 **Detector Settings**", reply_markup=detector_menu_kb(s))
        elif data == "menu:detector_pick":
            await cq.message.edit_text("Choose detector model:", reply_markup=enum_kb(DETECTORS, s["detector"], "set:detector", back="menu:detector"))

        elif data == "menu:ocr":
            await cq.message.edit_text("🔤 **OCR Settings**", reply_markup=ocr_menu_kb(s))
        elif data == "menu:ocr_pick":
            await cq.message.edit_text("Choose OCR model:", reply_markup=enum_kb(OCR_MODELS, s["ocr"], "set:ocr", back="menu:ocr"))

        elif data == "menu:translator":
            await cq.message.edit_text("🌐 **Translator Settings**", reply_markup=translator_menu_kb(s))
        elif data == "menu:translator_pick":
            await cq.message.edit_text("Choose translator engine:", reply_markup=enum_kb(TRANSLATORS, s["translator"], "set:translator", back="menu:translator", per_row=3))
        elif data == "menu:lang_pick":
            await cq.message.edit_text("Choose target language:", reply_markup=lang_kb(s["target_lang"]))

        elif data == "menu:custom_api":
            await cq.message.edit_text(
                "🔑 **Custom OpenAI-Compatible API**\n"
                "Used when Engine = `custom_openai`. Point this at any "
                "`/chat/completions`-compatible endpoint — OpenAI, Groq, Ollama, "
                "OpenRouter, a local server, etc.\n\n"
                "Tap a field below to type its value.",
                reply_markup=custom_api_menu_kb(s),
            )

        elif data.startswith("text:"):
            key = data.split(":", 1)[1]
            PENDING_TEXT_INPUT[user_id] = key
            await cq.answer()
            prompts = {
                "custom_openai_api_base": "Send the API base URL, e.g. `https://api.groq.com/openai/v1`",
                "custom_openai_model": "Send the model name, e.g. `gpt-4o-mini` or `qwen2.5:7b`",
                "custom_openai_api_key": "Send the API key (or `none` if the endpoint needs no key).",
            }
            await cq.message.reply_text(
                prompts.get(key, f"Send the new value for `{key}`.") + "\nSend `none` to clear it."
            )

        elif data == "menu:inpainter":
            await cq.message.edit_text("🎨 **Inpainter Settings**", reply_markup=inpainter_menu_kb(s))
        elif data == "menu:inpainter_pick":
            await cq.message.edit_text("Choose inpainter model:", reply_markup=enum_kb(INPAINTERS, s["inpainter"], "set:inpainter", back="menu:inpainter"))
        elif data == "menu:precision_pick":
            await cq.message.edit_text("Choose inpainting precision:", reply_markup=enum_kb(INPAINT_PRECISIONS, s["inpainting_precision"], "set:inpainting_precision", back="menu:inpainter"))

        elif data == "menu:colorizer":
            await cq.message.edit_text("🖌 **Colorizer Settings**", reply_markup=colorizer_menu_kb(s))
        elif data == "menu:colorizer_pick":
            await cq.message.edit_text("Choose colorizer model:", reply_markup=enum_kb(COLORIZERS, s["colorizer"], "set:colorizer", back="menu:colorizer"))

        elif data == "menu:upscaler":
            await cq.message.edit_text("📐 **Upscaler Settings**", reply_markup=upscaler_menu_kb(s))
        elif data == "menu:upscaler_pick":
            await cq.message.edit_text("Choose upscaler model:", reply_markup=enum_kb(UPSCALERS, s["upscaler"], "set:upscaler", back="menu:upscaler"))

        elif data == "menu:render":
            await cq.message.edit_text("✍️ **Render Settings**", reply_markup=render_menu_kb(s))
        elif data == "menu:renderer_pick":
            await cq.message.edit_text("Choose renderer:", reply_markup=enum_kb(RENDERERS, s["renderer"], "set:renderer", back="menu:render"))
        elif data == "menu:alignment_pick":
            await cq.message.edit_text("Choose alignment:", reply_markup=enum_kb(ALIGNMENTS, s["alignment"], "set:alignment", back="menu:render"))
        elif data == "menu:direction_pick":
            await cq.message.edit_text("Choose direction:", reply_markup=enum_kb(DIRECTIONS, s["direction"], "set:direction", back="menu:render"))

        elif data == "menu:font":
            fonts = get_user_fonts(user_id)
            await cq.message.edit_text(
                "🔤 **Font Settings**\n"
                "Choose a font to use for rendered text, or keep the repo's original/default font.\n"
                f"Uploaded fonts: `{len(fonts)}`",
                reply_markup=font_menu_kb(s, fonts),
            )

        elif data == "menu:font_add":
            PENDING_FONT_UPLOAD[user_id] = True
            await cq.answer()
            await cq.message.reply_text(
                "📤 Send me a `.ttf` or `.otf` font file now as a document.\n"
                "I'll add it to your font library. Use /cancel to abort."
            )

        elif data.startswith("selfont:"):
            font_id = data.split(":", 1)[1]
            f = get_user_font(user_id, font_id)
            if not f:
                await cq.answer("Font not found (maybe deleted).", show_alert=True)
            elif not f.get("repo_path"):
                await cq.answer("This font failed to sync to the repo — re-upload it.", show_alert=True)
            else:
                set_user_setting(user_id, "font_path", f["repo_path"])
                set_user_setting(user_id, "font_name", f["name"])
                await cq.answer(f"Font set to {f['name']}")
            s = get_user_settings(user_id)
            fonts = get_user_fonts(user_id)
            await cq.message.edit_reply_markup(reply_markup=font_menu_kb(s, fonts))

        elif data.startswith("delfont:"):
            font_id = data.split(":", 1)[1]
            ok = delete_user_font(user_id, font_id)
            await cq.answer("Font deleted" if ok else "Font not found", show_alert=not ok)
            s = get_user_settings(user_id)
            fonts = get_user_fonts(user_id)
            await cq.message.edit_text(
                "🔤 **Font Settings**\n"
                "Choose a font to use for rendered text, or keep the repo's original/default font.\n"
                f"Uploaded fonts: `{len(fonts)}`",
                reply_markup=font_menu_kb(s, fonts),
            )

        elif data == "menu:misc":
            await cq.message.edit_text("⚙️ **Misc Settings**", reply_markup=misc_menu_kb(s))

        elif data == "menu:humantl":
            await cq.message.edit_text(
                "👤 **Human-in-loop Translation Mode**\n"
                "When ON, the extraction step sends you the raw dialogue as a text file "
                "in PM and waits (max 10 min) for you to send back translations, instead "
                "of using the automatic translator engine.",
                reply_markup=humantl_kb(s),
            )

        elif data == "set:font_path:__default__":
            set_user_setting(user_id, "font_path", None)
            set_user_setting(user_id, "font_name", None)
            await cq.answer("Font reset to Original (Default)")
            s = get_user_settings(user_id)
            fonts = get_user_fonts(user_id)
            await cq.message.edit_reply_markup(reply_markup=font_menu_kb(s, fonts))

        elif data.startswith("set:"):
            _, key, value = data.split(":", 2)
            if value == "true":
                value = True
            elif value == "false":
                value = False
            set_user_setting(user_id, key, value)
            await cq.answer(f"{key} = {value}")
            # Re-render the menu it came from based on key
            s = get_user_settings(user_id)
            back_map = {
                "detector": detector_menu_kb, "ocr": ocr_menu_kb, "translator": translator_menu_kb,
                "target_lang": translator_menu_kb, "inpainter": inpainter_menu_kb,
                "inpainting_precision": inpainter_menu_kb, "colorizer": colorizer_menu_kb,
                "upscaler": upscaler_menu_kb, "renderer": render_menu_kb,
                "alignment": render_menu_kb, "direction": render_menu_kb,
            }
            kb_fn = back_map.get(key, main_settings_kb)
            try:
                await cq.message.edit_reply_markup(reply_markup=kb_fn(s) if kb_fn != main_settings_kb else main_settings_kb())
            except Exception:
                pass

        elif data.startswith("toggle:"):
            _, key = data.split(":", 1)
            new_val = not bool(s.get(key, False))
            set_user_setting(user_id, key, new_val)
            await cq.answer(f"{key} = {new_val}")
            s = get_user_settings(user_id)
            if key in ("det_rotate", "det_auto_rotate", "det_invert", "det_gamma_correct"):
                await cq.message.edit_reply_markup(reply_markup=misc_menu_kb(s) if cq.message.text and "Misc" in cq.message.text else detector_menu_kb(s))
            elif key in ("uppercase", "lowercase", "disable_font_border", "no_hyphenation", "rtl"):
                await cq.message.edit_reply_markup(reply_markup=render_menu_kb(s))
            elif key == "use_mocr_merge":
                await cq.message.edit_reply_markup(reply_markup=ocr_menu_kb(s))
            elif key == "no_text_lang_skip":
                await cq.message.edit_reply_markup(reply_markup=translator_menu_kb(s))
            elif key == "revert_upscaling":
                await cq.message.edit_reply_markup(reply_markup=upscaler_menu_kb(s))
            elif key == "human_translation_mode":
                await cq.message.edit_text(
                    "👤 **Human-in-loop Translation Mode**\n"
                    "When ON, the extraction step sends you the raw dialogue as a text file "
                    "in PM and waits (max 10 min) for you to send back translations, instead "
                    "of using the automatic translator engine.",
                    reply_markup=humantl_kb(s),
                )
            else:
                await cq.message.edit_reply_markup(reply_markup=main_settings_kb())

        elif data.startswith("numeric:"):
            _, key = data.split(":", 1)
            PENDING_NUMERIC_INPUT[user_id] = key
            await cq.answer()
            await cq.message.reply_text(
                f"Send the new numeric value for `{key}` (current: `{s.get(key)}`).\n"
                f"Send `none` to clear it (for optional fields)."
            )

        else:
            await cq.answer()

    except Exception as e:
        log.exception("callback handling failed")
        try:
            await cq.answer(f"Error: {e}", show_alert=True)
        except Exception:
            pass


@bot.on_message(filters.private & filters.text & ~filters.command(
    ["start", "settings", "translate", "cancel", "myjobs", "status"]
))
async def on_text_reply(_, message: Message):
    user_id = message.from_user.id

    text_key = PENDING_TEXT_INPUT.get(user_id)
    if text_key:
        PENDING_TEXT_INPUT.pop(user_id, None)
        raw = message.text.strip()
        if raw.lower() == "none":
            set_user_setting(user_id, text_key, None)
            await message.reply_text(f"✅ `{text_key}` cleared.")
        else:
            set_user_setting(user_id, text_key, raw)
            shown = raw if text_key != "custom_openai_api_key" else "•••• (saved)"
            await message.reply_text(f"✅ `{text_key}` = `{shown}`")
        return

    key = PENDING_NUMERIC_INPUT.get(user_id)
    if not key:
        return  # not expecting numeric input; ignore

    raw = message.text.strip()
    PENDING_NUMERIC_INPUT.pop(user_id, None)

    if raw.lower() == "none":
        set_user_setting(user_id, key, None)
        await message.reply_text(f"✅ `{key}` cleared.")
        return

    try:
        value = float(raw) if "." in raw else int(raw)
    except ValueError:
        await message.reply_text("❌ That's not a valid number. Setting unchanged.")
        return

    set_user_setting(user_id, key, value)
    await message.reply_text(f"✅ `{key}` = `{value}`")


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    log.info("Starting Tamatar-Laal bot...")
    bot.run()

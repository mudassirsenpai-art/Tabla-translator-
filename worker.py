"""
Tamatar-Laal Manga Translator — GitHub Actions Worker
=======================================================
Triggered by bot.py via workflow_dispatch. Downloads the file from Telegram,
runs manga-image-translator with the FULL config the user picked in /settings,
posts live progress + a final JSON result back to Telegram, then cleans up.

All secrets come from environment variables / GitHub Actions secrets — nothing
is hardcoded in this file.
"""

import os
import sys
import json
import zipfile
import shutil
import asyncio
import time
import requests
from typing import Optional
from pyrogram import Client
import pyrogram.utils

pyrogram.utils.get_peer_type = lambda p: (
    "channel" if str(p).startswith("-100") else "chat" if str(p).startswith("-") else "user"
)

# ============================================================================
# INPUTS (from workflow_dispatch inputs / repo secrets — no literals here)
# ============================================================================
JOB_ID = os.getenv("JOB_ID", "").strip()
FILE_ID = os.getenv("FILE_ID", "").strip()
CHAT_ID = int(os.getenv("CHAT_ID", "0"))
MSG_ID = int(os.getenv("MSG_ID", "0"))
USER_ID = int(os.getenv("USER_ID", "0"))
FNAME = os.getenv("FNAME", "manga.zip").strip()
CONFIG_JSON = os.getenv("CONFIG_JSON", "{}").strip()

API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "").strip()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
REPO_NAME = os.getenv("REPO_NAME", "").strip()

if not (API_ID and API_HASH and BOT_TOKEN and REPO_NAME):
    print("Missing required secrets/env vars (API_ID, API_HASH, BOT_TOKEN, REPO_NAME). Exiting.")
    sys.exit(1)

try:
    CONFIG = json.loads(CONFIG_JSON)
except Exception:
    CONFIG = {}

# CPU tuning
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# --------------------------------------------------------------------------
# custom_openai translator: user-provided endpoint/model/key (set via
# /settings in bot.py) take priority; repo-level secrets act as a fallback
# default so a job still works if the user didn't override anything.
# --------------------------------------------------------------------------
_custom_base = CONFIG.get("custom_openai_api_base") or os.getenv("CUSTOM_OPENAI_API_BASE", "")
_custom_model = CONFIG.get("custom_openai_model") or os.getenv("CUSTOM_OPENAI_MODEL", "")
_custom_key = CONFIG.get("custom_openai_api_key") or os.getenv("CUSTOM_OPENAI_API_KEY", "")
if _custom_base:
    os.environ["CUSTOM_OPENAI_API_BASE"] = _custom_base
if _custom_model:
    os.environ["CUSTOM_OPENAI_MODEL"] = _custom_model
if _custom_key:
    os.environ["CUSTOM_OPENAI_API_KEY"] = _custom_key
try:
    import torch
    torch.set_num_threads(4)
except ImportError:
    pass


# ============================================================================
# CONFIG -> manga_translator CLI flag builder
# ============================================================================

def build_cli_flags(cfg: dict, target_lang: str) -> list:
    flags = ["-l", target_lang]

    def add(flag, key, cast=str):
        if key in cfg and cfg[key] is not None:
            flags.extend([flag, str(cast(cfg[key]))])

    def add_bool(flag, key):
        if cfg.get(key):
            flags.append(flag)

    # Detector
    add("--detector", "detector")
    add("--detection-size", "detection_size", int)
    add("--text-threshold", "text_threshold", float)
    add_bool("--det-rotate", "det_rotate")
    add_bool("--det-auto-rotate", "det_auto_rotate")
    add_bool("--det-invert", "det_invert")
    add_bool("--det-gamma-correct", "det_gamma_correct")
    add("--box-threshold", "box_threshold", float)
    add("--unclip-ratio", "unclip_ratio", float)

    # OCR
    add("--ocr", "ocr")
    add("--min-text-length", "min_text_length", int)
    add("--ignore-bubble", "ignore_bubble", int)

    # Inpainter
    add("--inpainter", "inpainter")
    add("--inpainting-size", "inpainting_size", int)
    add("--inpainting-precision", "inpainting_precision")

    # Colorizer
    if cfg.get("colorizer") and cfg["colorizer"] != "none":
        add("--colorizer", "colorizer")
        add("--colorization-size", "colorization_size", int)
        add("--denoise-sigma", "denoise_sigma", int)

    # Upscaler
    if cfg.get("upscale_ratio"):
        add("--upscaler", "upscaler")
        add("--upscale-ratio", "upscale_ratio", int)
        add_bool("--revert-upscaling", "revert_upscaling")

    # Translator
    add("--translator", "translator")
    add_bool("--no-text-lang-skip", "no_text_lang_skip")

    # Render
    if cfg.get("renderer") == "manga2eng":
        flags.append("--manga2eng")
    add("--alignment", "alignment")
    add("--direction", "direction")
    add_bool("--disable-font-border", "disable_font_border")
    add("--font-size-offset", "font_size_offset", int)
    add("--font-size-minimum", "font_size_minimum", int)
    add_bool("--uppercase", "uppercase")
    add_bool("--lowercase", "lowercase")
    add_bool("--no-hyphenation", "no_hyphenation")
    add("--font-path", "_resolved_font_path")

    # Misc
    add("--kernel-size", "kernel_size", int)
    add("--mask-dilation-offset", "mask_dilation_offset", int)

    return flags


def resolve_font_path(cfg: dict) -> Optional[str]:
    """
    cfg['font_path'], if set, is a path *inside this bot's repo* (e.g.
    'user_fonts/12345/abcd.ttf'), because the font was uploaded via Telegram
    to bot.py and pushed there so the (separate) Actions runner can see it.
    Download it here to a local temp file and return that local path, or
    None if unset / not original-default / download fails (falls back to
    the engine's default font).
    """
    repo_font_path = cfg.get("font_path")
    if not repo_font_path:
        return None  # "Original (Default)" — no --font-path passed

    headers = {"Accept": "application/vnd.github.v3.raw", "Authorization": f"token {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{REPO_NAME}/contents/{repo_font_path}"
    try:
        res = requests.get(url, headers=headers, timeout=20)
        if res.status_code != 200:
            print(f"Font download failed ({res.status_code}) for {repo_font_path} — using default font.")
            return None
        local_path = os.path.abspath("downloaded_font" + os.path.splitext(repo_font_path)[1])
        with open(local_path, "wb") as f:
            f.write(res.content)
        return local_path
    except Exception as e:
        print("Font download exception:", e, "— using default font.")
        return None


async def edit_status(bot_client, text):
    try:
        await bot_client.edit_message_text(CHAT_ID, MSG_ID, text)
    except Exception as e:
        print("Status edit failed:", e)


def progress_bar(percent):
    filled = int(percent // 10)
    return "█" * filled + "░" * (10 - filled)


async def run_pipeline(input_dir, output_dir, bot_client):
    cwd_dir = "manga-image-translator" if os.path.exists("manga-image-translator") else None

    target_lang = CONFIG.get("target_lang", "ENG")
    resolved_font = resolve_font_path(CONFIG)
    if resolved_font:
        CONFIG["_resolved_font_path"] = resolved_font
    cli_flags = build_cli_flags(CONFIG, target_lang)
    human_mode = bool(CONFIG.get("human_translation_mode", False))

    pages = sorted([
        os.path.join(r, f) for r, _, fs in os.walk(input_dir) for f in fs
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    ])
    if not pages:
        return False, "No pages found", {}

    result_summary = {
        "job_id": JOB_ID,
        "pages_total": len(pages),
        "target_lang": target_lang,
        "config": CONFIG,
        "human_translation_mode": human_mode,
        "phases": [],
    }

    if human_mode:
        ok, msg, extra = await run_human_translation_flow(input_dir, output_dir, cwd_dir, cli_flags, pages, bot_client)
        result_summary["phases"].append({"phase": "human_translation", "ok": ok, "detail": msg})
        result_summary.update(extra)
        return ok, msg, result_summary

    # -------------------- Fully automatic mode --------------------
    cmd = ["python", "-m", "manga_translator", "local", "-i", input_dir, "-o", output_dir] + cli_flags
    await edit_status(bot_client, f"🔍 **Translating manga...**\n`[{progress_bar(20)}] 20%`")

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=cwd_dir
    )

    current_page = 0
    start_time = time.time()
    log_tail = []

    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="ignore").strip()
        print(decoded)
        log_tail.append(decoded)
        log_tail = log_tail[-50:]

        if "Translating:" in decoded:
            current_page += 1
            elapsed = time.time() - start_time
            speed = current_page / elapsed if elapsed > 0 else 0
            percent = int((current_page / max(len(pages), 1)) * 100)
            speed_str = f"{speed:.2f} pages/sec"
            await edit_status(
                bot_client,
                f"🔍 **Translating manga**\n"
                f"Page `{current_page}`/`{len(pages)}` — {speed_str}\n"
                f"`[{progress_bar(percent)}] {percent}%`",
            )

    await proc.wait()

    cnt_results = 0
    if os.path.exists(output_dir):
        cnt_results = len([
            f for r, _, fx in os.walk(output_dir) for f in fx
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])

    result_summary["pages_translated"] = cnt_results
    result_summary["returncode"] = proc.returncode
    result_summary["log_tail"] = log_tail

    if proc.returncode == 0 and cnt_results > 0:
        return True, "Success", result_summary
    return False, "manga_translator failed", result_summary


async def run_human_translation_flow(input_dir, output_dir, cwd_dir, cli_flags, pages, bot_client):
    """
    Optional flow: OCR-extract dialogue -> send to user PM as a text file ->
    wait up to 10 min for them to reply with translations -> render.
    """
    ws = os.path.abspath("manga_workspace")

    bypass_code = '''
import os, json
from .common import CommonTranslator

class HumanInterventionTranslator(CommonTranslator):
    supported_src_languages = ["auto"]
    supported_target_languages = ["auto"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.frame_counter = 0
        self.translations_map = {}
        mode = os.environ.get("ENV_TRANSLATE_MODE", "EXTRACT")
        if mode == "RENDER":
            p = "../manga_workspace/translations.json"
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    pi, bi = k.split("_", 1)
                    self.translations_map[(int(pi), bi.strip())] = v

    def supports_languages(self, f, t, fatal=False):
        return True

    async def _translate(self, f, t, queries, *a, **kw):
        return await self.do_custom_workflow(queries)

    async def translate(self, f, t, queries, *a, **kw):
        return await self.do_custom_workflow(queries)

    async def do_custom_workflow(self, queries):
        if not queries:
            return queries
        self.frame_counter += 1
        mode = os.environ.get("ENV_TRANSLATE_MODE", "EXTRACT")
        if mode == "EXTRACT":
            os.makedirs("../manga_workspace", exist_ok=True)
            with open(f"../manga_workspace/page_{self.frame_counter}_queries.txt", "w", encoding="utf-8") as f:
                f.write("\\n".join(queries))
            return queries
        elif mode == "RENDER":
            out = []
            for idx, q in enumerate(queries, 1):
                key = (self.frame_counter, str(idx))
                out.append(self.translations_map.get(key, q))
            return out

class ChatGPTTranslator(HumanInterventionTranslator): pass
class ChatGPT2StageTranslator(HumanInterventionTranslator): pass
class GPT3Translator(HumanInterventionTranslator): pass
class GPT35TurboTranslator(HumanInterventionTranslator): pass
class GPT4Translator(HumanInterventionTranslator): pass
'''
    if cwd_dir:
        node = os.path.join(cwd_dir, "manga_translator", "translators", "chatgpt.py")
        if os.path.exists(os.path.dirname(node)):
            with open(node, "w", encoding="utf-8") as f:
                f.write(bypass_code)

    extract_flags = ["--translator", "gpt3"]
    skip_next = False
    for i, x in enumerate(cli_flags):
        if skip_next:
            skip_next = False
            continue
        if x == "--translator":
            skip_next = True
            continue
        extract_flags.append(x)

    os.environ["ENV_TRANSLATE_MODE"] = "EXTRACT"
    cmd = ["python", "-m", "manga_translator", "local", "-i", input_dir, "-o", output_dir] + extract_flags
    await edit_status(bot_client, f"🔍 **Phase 1/3: Extracting dialogue (OCR)**\n`[{progress_bar(30)}] 30%`")

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=cwd_dir)
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(line.decode("utf-8", errors="ignore").strip())
    await proc.wait()

    master_lines = []
    for i in range(1, 1000):
        pf = os.path.join(ws, f"page_{i}_queries.txt")
        if not os.path.exists(pf):
            if i > 1 and not any(os.path.exists(os.path.join(ws, f"page_{k}_queries.txt")) for k in range(i, i + 10)):
                break
            continue
        master_lines.append(f"[Page {i:02d}]")
        with open(pf, "r", encoding="utf-8") as f:
            queries = f.read().splitlines()
        for idx, q in enumerate(queries, 1):
            master_lines.append(str(idx))
            master_lines.append(f"{{{USER_ID}}}tutty_{i}_{idx} ==> {q}\n")
        master_lines.append("")

    master_txt_path = os.path.join(ws, f"FrameExtr_{USER_ID}.txt")
    with open(master_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(master_lines))

    caption_pm = (
        "📝 **Manga dialogue extracted — translate & send back**\n\n"
        f"Pages: `{len(pages)}`\n"
        "Translate the text after `==>`. Do NOT alter the tags or arrow.\n"
        "You have **10 minutes** to send this file back to the bot in PM."
    )
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    try:
        with open(master_txt_path, "rb") as doc:
            requests.post(url, data={"chat_id": USER_ID, "caption": caption_pm}, files={"document": doc}, timeout=15)
    except Exception as e:
        print("Failed to deliver extraction doc:", e)

    timeout_duration = 600
    start_time = time.time()
    user_uploaded = False
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}

    while time.time() - start_time < timeout_duration:
        remaining = timeout_duration - int(time.time() - start_time)
        mins, secs = divmod(remaining, 60)
        percent = int(((timeout_duration - remaining) / timeout_duration) * 100)
        await edit_status(
            bot_client,
            f"⏳ **Phase 2/3: Waiting for your translation**\n"
            f"Time left: `{mins:02d}m {secs:02d}s`\n`[{progress_bar(percent)}] {percent}%`",
        )
        api_url = f"https://api.github.com/repos/{REPO_NAME}/contents/trans_{USER_ID}.txt?t={int(time.time())}"
        try:
            res = requests.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                import base64, re
                data = res.json()
                txt_val = base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")
                pattern = r"\{(\d+)\}tutty_(\d+)_(\d+) ==> (.*)"
                translations = {}
                for line in txt_val.splitlines():
                    m = re.search(pattern, line.strip())
                    if m:
                        _, p_idx, b_idx, text = m.groups()
                        translations[f"{p_idx}_{b_idx}"] = text.strip()
                with open(os.path.join(ws, "translations.json"), "w", encoding="utf-8") as f:
                    json.dump(translations, f, ensure_ascii=False, indent=2)
                sha = data.get("sha")
                requests.delete(api_url.split("?")[0], headers=headers, json={"message": "cleanup", "sha": sha}, timeout=10)
                user_uploaded = True
                break
        except Exception as e:
            print("Poller error:", e)
        await asyncio.sleep(10)

    if not user_uploaded:
        return False, "Timeout waiting for user translation", {"timed_out": True}

    os.environ["ENV_TRANSLATE_MODE"] = "RENDER"
    await edit_status(bot_client, f"🎨 **Phase 3/3: Rendering**\n`[{progress_bar(80)}] 80%`")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmd2 = ["python", "-m", "manga_translator", "local", "-i", input_dir, "-o", output_dir] + extract_flags
    proc2 = await asyncio.create_subprocess_exec(*cmd2, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=cwd_dir)
    while True:
        line = await proc2.stdout.readline()
        if not line:
            break
        print(line.decode("utf-8", errors="ignore").strip())
    await proc2.wait()

    cnt_results = 0
    if os.path.exists(output_dir):
        cnt_results = len([f for r, _, fx in os.walk(output_dir) for f in fx if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])

    if proc2.returncode == 0 and cnt_results > 0:
        return True, "Success", {"pages_translated": cnt_results}
    return False, "Render failed", {"pages_translated": cnt_results}


# ============================================================================
# MAIN
# ============================================================================

async def main():
    result = {"job_id": JOB_ID, "user_id": USER_ID, "chat_id": CHAT_ID, "started_at": int(time.time())}

    if not FILE_ID:
        print("Empty FILE_ID, exiting.")
        result["error"] = "empty file_id"
        print(json.dumps(result))
        return

    tg_bot = Client("WorkerMaster", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN, no_updates=True)
    await tg_bot.start()

    await edit_status(tg_bot, f"⚡ **Job started**\n`[{progress_bar(10)}] 10%`")

    dl_path = None
    for attempt in range(1, 6):
        try:
            dl_path = await tg_bot.download_media(FILE_ID)
            if dl_path and os.path.exists(dl_path) and os.path.getsize(dl_path) > 1024:
                break
            await asyncio.sleep(2)
        except Exception as e:
            await edit_status(tg_bot, f"⚠️ **Network retry {attempt}/5:** `{e}`")
            await asyncio.sleep(3)

    if not dl_path or not os.path.exists(dl_path):
        result["error"] = "download failed"
        await edit_status(tg_bot, f"❌ **Failed to download file**\n```json\n{json.dumps(result)}\n```")
        await tg_bot.stop()
        return

    await edit_status(tg_bot, f"📦 **Extracting...**\n`[{progress_bar(20)}] 20%`")

    ext = os.path.splitext(FNAME)[1].lower() or ".zip"
    ws = os.path.abspath("manga_workspace")
    inp = os.path.join(ws, "input")
    out = os.path.join(ws, "output")
    if os.path.exists(ws):
        shutil.rmtree(ws)
    os.makedirs(inp, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    try:
        if ext in (".zip", ".cbz"):
            with zipfile.ZipFile(dl_path, "r") as z:
                z.extractall(inp)
        elif ext == ".pdf":
            import fitz
            pdf = fitz.open(dl_path)
            for i in range(len(pdf)):
                pdf.load_page(i).get_pixmap(dpi=150).save(os.path.join(inp, f"page_{i:03d}.png"))
            pdf.close()
        else:
            shutil.copy(dl_path, inp)
    except Exception as e:
        result["error"] = f"extraction failed: {e}"
        await edit_status(tg_bot, f"❌ **Extraction failed**\n```json\n{json.dumps(result)}\n```")
        await tg_bot.stop()
        return

    ok, msg, summary = await run_pipeline(inp, out, tg_bot)
    result.update(summary if isinstance(summary, dict) else {})
    result["ok"] = ok
    result["message"] = msg
    result["finished_at"] = int(time.time())

    if not ok:
        await edit_status(
            tg_bot,
            f"❌ **Job failed**\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)[:3500]}\n```",
        )
        await tg_bot.stop()
        return

    await edit_status(tg_bot, f"📦 **Packaging result...**\n`[{progress_bar(90)}] 90%`")

    finals = sorted([
        os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])
    zip_out = "translated_" + FNAME if ext in (".zip", ".cbz", ".pdf") else (finals[0] if finals else None)

    try:
        if ext in (".zip", ".cbz"):
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for fpath in finals:
                    z.write(fpath, os.path.relpath(fpath, out))
        elif ext == ".pdf":
            import fitz
            doc = fitz.open()
            for img_path in finals:
                img = fitz.open(img_path)
                rect = img[0].rect
                page = doc.new_page(width=rect.width, height=rect.height)
                page.insert_image(rect, filename=img_path, keep_proportion=True)
                img.close()
            doc.save(zip_out, garbage=4, deflate=True)
            doc.close()
    except Exception as e:
        result["error"] = f"packaging failed: {e}"
        await edit_status(tg_bot, f"❌ **Packaging failed**\n```json\n{json.dumps(result, indent=2)[:3500]}\n```")
        await tg_bot.stop()
        return

    try:
        if zip_out:
            await tg_bot.send_document(
                CHAT_ID, zip_out,
                caption=f"✅ **Done!** Job `{JOB_ID}`\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)[:900]}\n```",
            )
        await tg_bot.delete_messages(CHAT_ID, MSG_ID)
    except Exception as e:
        print("Failed to deliver final document:", e)

    # cleanup
    shutil.rmtree(ws, ignore_errors=True)
    try:
        os.remove(dl_path)
    except Exception:
        pass
    try:
        if zip_out and ext in (".zip", ".cbz", ".pdf"):
            os.remove(zip_out)
    except Exception:
        pass

    print(json.dumps(result, ensure_ascii=False))
    await tg_bot.stop()


if __name__ == "__main__":
    asyncio.run(main())

# app.py
# Multimodal Chatbot — Two-Pane Layout: Chat + Media on Left, Text-to-Image Studio on Right
# -----------------------------------------------------------------------------
# Changes in this version:
# - API keys always masked (no eye reveal) and sourced from st.secrets by default.
# - Optional user override fields are masked and never show the actual key; no save/clear/test buttons.
# - No environment writes; providers read from st.secrets or an in-session override only.

import os
import io
import time
import json
import random
import pathlib
from typing import Tuple, List

import streamlit as st
from dotenv import load_dotenv

# Provider SDKs
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from groq import Groq
from tavily import TavilyClient  # NEW

# Standard utils
from PIL import Image

# =============================================================================
# Environment
# =============================================================================
load_dotenv()

# =============================================================================
# Persistent App Storage Helpers (JSON file in user home)
# =============================================================================
APP_DATA_DIR = pathlib.Path.home() / ".multimodal_chatbot_app"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = APP_DATA_DIR / "memory.json"

def load_memory() -> dict:
    try:
        if MEMORY_FILE.exists():
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_memory(data: dict):
    try:
        MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

GLOBAL_MEMORY = load_memory()

def remember_user_name(name: str):
    GLOBAL_MEMORY["user_name"] = name.strip()
    save_memory(GLOBAL_MEMORY)

def recall_user_name() -> str:
    return GLOBAL_MEMORY.get("user_name", "")

# =============================================================================
# Page config & Theming
# =============================================================================
st.set_page_config(
    page_title="Multimodal Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS: keep existing + hide password reveal eye in text_input
st.markdown("""
<style>
/* Tighten file uploader and audio inputs */
div[data-testid="stFileUploader"] > section { padding: 6px 8px !important; }
div[data-testid="stFileUploader"] label { font-size: 0.85rem !important; }
div[data-testid="stFileUploader"] .uploadedFile { margin-top: 4px !important; }
div[data-testid="stAudioInput"] { padding: 6px 8px !important; }
/* Left pane: chat+media column sizing helpers */
.left-pane { max-width: 100%; }
.media-card { padding: 10px 12px; }
.media-narrow { max-width: 420px; }
/* Right pane: Text-to-Image studio card */
.tii-card {
  border: 1px solid rgba(250,250,250,0.10);
  border-radius: 10px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}
.tii-card h3 { margin: 0 0 8px 0; font-weight: 600; }
.tii-subtle { color: rgba(250,250,250,0.7); font-size: 0.9rem; margin-bottom: 6px; }
/* Loader surface: compact font + scrollable if needed */
.loader-wrap {
  border: 1px solid rgba(250,250,250,0.08);
  border-radius: 8px;
  padding: 6px 10px;
  background: rgba(255,255,255,0.02);
  font-size: 0.9rem;
  line-height: 1.35;
  max-height: 140px;
  overflow-y: auto;
}
/* Hide Streamlit's "show password" eye control */
[title="Show password text"], [title="Hide password text"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State
# =============================================================================
def _get_ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

_get_ss("messages", [])
# Remove env dependence; seed with st.secrets defaults, allow in-session override
_get_ss("openai_api_key", st.secrets.get("OPENAI_API_KEY", ""))
_get_ss("gemini_api_key", st.secrets.get("GEMINI_API_KEY", st.secrets.get("GEMINI_KEY", "")))
_get_ss("groq_api_key", st.secrets.get("GROQ_API_KEY", ""))
_get_ss("tavily_api_key", st.secrets.get("TAVILY_API_KEY", ""))
_get_ss("selected_model", "Gemini - (gemini-2.5-flash)")
_get_ss("enable_router", True)
_get_ss("enable_web_router", True)
_get_ss("stt_backend", "OpenAI Whisper")
_get_ss("last_error", "")
_get_ss("generated_images", [])
_get_ss("image_qa_mode", False)
_get_ss("tii_prompt", "")
_get_ss("user_name", recall_user_name())
_get_ss("web_used", False)

# =============================================================================
# UI helpers
# =============================================================================
def show_success(msg: str):
    st.success(msg)

def show_warn(msg: str):
    st.warning(msg)
    st.session_state["last_error"] = msg

def show_error(msg: str):
    st.error(msg)
    st.session_state["last_error"] = msg

# Minimal, professional loader helper
def minimal_loader_text(kind: str) -> str:
    mapping = {"thinking": "Thinking…", "generating": "Generating…", "loading": "Loading…"}
    return mapping.get(kind, "Loading…")

def rotating_spinner_content(ph, run_callable, kind="thinking"):
    result = {"val": None, "err": None}
    with ph.container():
        st.markdown('<div class="loader-wrap">', unsafe_allow_html=True)
        msg_ph = st.empty()
        msg_ph.info(minimal_loader_text(kind))
        try:
            try:
                result["val"] = run_callable()
            except Exception as e:
                result["err"] = e
        finally:
            st.markdown('</div>', unsafe_allow_html=True)
    if result["err"]:
        ph.error(f"Request failed: {result['err']}")
        raise result["err"]
    ph.markdown(result["val"])
    return result["val"]

# =============================================================================
# Model config
# =============================================================================
MODEL_OPTIONS = (
    "Gemini - (gemini-2.5-flash)",
    "OpenAI - (gpt-4o-mini)",
    "groq - (llama-3.3-70b-versatile)",
)

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("Controls")
    st.caption("Switch models, manage keys, and enable features.")

    try:
        default_index = MODEL_OPTIONS.index(st.session_state["selected_model"])
    except ValueError:
        default_index = 0
    st.session_state["selected_model"] = st.selectbox(
        "Model",
        MODEL_OPTIONS,
        index=default_index,
        help="Switch between providers at any time.",
    )
    st.divider()

    # API Keys (masked overwrite only; defaults come from st.secrets)
    st.subheader("API Keys")
    st.caption("Keys default from st.secrets. Optionally overwrite for this session. Value stays masked.")

    with st.expander("Gemini", expanded=False):
        new_gem = st.text_input("Gemini API Key (session override)", type="password", value="", placeholder="••••••••")
        if new_gem:
            st.session_state["gemini_api_key"] = new_gem
            show_success("Gemini key updated for this session.")

    with st.expander("OpenAI", expanded=False):
        new_oa = st.text_input("OpenAI API Key (session override)", type="password", value="", placeholder="••••••••")
        if new_oa:
            st.session_state["openai_api_key"] = new_oa
            show_success("OpenAI key updated for this session.")

    with st.expander("Groq", expanded=False):
        new_gq = st.text_input("Groq API Key (session override)", type="password", value="", placeholder="••••••••")
        if new_gq:
            st.session_state["groq_api_key"] = new_gq
            show_success("Groq key updated for this session.")

    with st.expander("Tavily", expanded=False):
        new_tv = st.text_input("Tavily API Key (session override)", type="password", value="", placeholder="••••••••")
        if new_tv:
            st.session_state["tavily_api_key"] = new_tv
            show_success("Tavily key updated for this session.")

    st.divider()
    st.subheader("Features")
    st.session_state["enable_router"] = st.checkbox("Enable prompt router", value=st.session_state["enable_router"])
    st.session_state["stt_backend"] = st.selectbox(
        "Voice-to-Text",
        ["OpenAI Whisper", "Gemini STT"],
        index=(0 if st.session_state["stt_backend"] == "OpenAI Whisper" else 1)
    )
    st.session_state["image_qa_mode"] = st.toggle("Ask about uploaded image", value=st.session_state["image_qa_mode"])
    st.session_state["enable_web_router"] = st.checkbox("Enable web query checker (Tavily)", value=st.session_state["enable_web_router"])

    st.divider()
    st.subheader("Quick actions")
    qa1, qa2 = st.columns(2)
    if qa1.button("Clear Chat"):
        st.session_state["messages"] = []
        show_success("Chat cleared.")
    if qa2.button("Reset Errors"):
        st.session_state["last_error"] = ""
        show_success("Errors reset.")

    st.divider()
    st.subheader("Identity")
    new_name = st.text_input("Your name (remembered)", value=st.session_state["user_name"])
    if new_name != st.session_state["user_name"]:
        st.session_state["user_name"] = new_name.strip()
        if st.session_state["user_name"]:
            remember_user_name(st.session_state["user_name"])
            show_success("Name saved to memory.")

    st.divider()
    st.subheader("Diagnostics")
    if st.session_state["last_error"]:
        st.caption(f"Last error: {st.session_state['last_error']}")

# =============================================================================
# Router/Polisher
# =============================================================================
def route_and_polish(user_text: str) -> Tuple[str, str]:
    system = (
        "You are a helpful, precise assistant. Reply concisely, use Markdown when helpful, "
        "and give direct answers. Provide runnable code with comments when asked for code."
    )
    if st.session_state.get("user_name"):
        system += f" The user is named {st.session_state['user_name']}."
        system += " Always remember to address the user by their name if appropriate."
    if len(user_text.strip()) < 8:
        system += " The user query is short; infer likely intent and provide a useful answer."
    uname = st.session_state.get("user_name", "").strip()
    lower_text = user_text.lower()
    if uname and any(k in lower_text for k in [
        "your name", "do you know my name", "remember my name", "what is my name", "tell me my name"
    ]):
        return system, f"Your name is {uname}."
    return system, user_text.strip()

# =============================================================================
# Providers
# =============================================================================
def _require_key(val: str, label: str):
    if not val:
        raise RuntimeError(f"{label} missing. Provide it via st.secrets or sidebar override.")
    return val

def gemini_model_function(prompt: str) -> str:
    try:
        api_key = st.session_state["gemini_api_key"] or st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
        api_key = _require_key(api_key, "Gemini API key")
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = getattr(resp, "text", "") or ""
        if not text:
            for cand in getattr(resp, "candidates", []):
                content = getattr(cand, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        if getattr(part, "text", None):
                            text += part.text
        if not text:
            raise RuntimeError("Empty response from Gemini.")
        return text
    except Exception as e:
        show_error(f"Gemini error: {e}")
        raise

def openai_model_function(prompt: str) -> str:
    try:
        api_key = st.session_state["openai_api_key"] or st.secrets.get("OPENAI_API_KEY")
        api_key = _require_key(api_key, "OpenAI API key")
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            text={"format": {"type": "text"}},
            max_output_tokens=2048,
        )
        out = getattr(response, "output_text", "") or ""
        if not out:
            try:
                for item in getattr(response, "output", []):
                    if getattr(item, "type", "") == "message":
                        for c in getattr(item, "content", []):
                            if getattr(c, "type", "") == "output_text":
                                out += getattr(c, "text", "")
            except Exception:
                pass
        if not out:
            raise RuntimeError("Empty response from OpenAI.")
        return out
    except Exception as e:
        show_error(f"OpenAI error: {e}")
        raise

def groq_model_function(prompt: str) -> str:
    try:
        api_key = st.session_state["groq_api_key"] or st.secrets.get("GROQ_API_KEY")
        api_key = _require_key(api_key, "Groq API key")
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
        )
        text = ""
        if chat_completion and getattr(chat_completion, "choices", None):
            choice0 = chat_completion.choices if len(chat_completion.choices) > 0 else None
            if choice0 and getattr(choice0, "message", None):
                text = getattr(choice0.message, "content", "") or ""
        if not text:
            raise RuntimeError("Empty response from Groq.")
        return text
    except Exception as e:
        show_error(f"Groq error: {e}")
        raise

def run_model(selected_model: str, prompt: str) -> str:
    if selected_model.startswith("Gemini"):
        return gemini_model_function(prompt)
    if selected_model.startswith("OpenAI"):
        return openai_model_function(prompt)
    if selected_model.startswith("groq"):
        return groq_model_function(prompt)
    raise RuntimeError("Unknown model selected.")

# =============================================================================
# STT
# =============================================================================
def stt_openai(audio_bytes: bytes) -> str:
    try:
        api_key = st.session_state["openai_api_key"] or st.secrets.get("OPENAI_API_KEY")
        api_key = _require_key(api_key, "OpenAI API key (STT)")
        client = OpenAI(api_key=api_key)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            path = f.name
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(path, "rb")
        )
        return getattr(transcription, "text", "") or ""
    except Exception as e:
        show_error(f"OpenAI STT error: {e}")
        return ""

def stt_gemini(audio_bytes: bytes) -> str:
    try:
        api_key = st.session_state["gemini_api_key"] or st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
        api_key = _require_key(api_key, "Gemini API key (STT)")
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"role": "user", "parts": [
                    {"text": "Transcribe this audio faithfully and return plain text."},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}}
                ]}
            ]
        )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        show_error(f"Gemini STT error: {e}")
        return ""

# =============================================================================
# Vision
# =============================================================================
def gemini_vision_answer(question: str, image_file) -> str:
    try:
        api_key = st.session_state["gemini_api_key"] or st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_KEY")
        api_key = _require_key(api_key, "Gemini API key")
        if image_file is None:
            raise RuntimeError("No image uploaded to analyze.")
        img_bytes = image_file.read()
        mime = f"image/{(image_file.type or 'png').split('/')[-1]}"
        client = genai.Client(api_key=api_key)
        contents = [
            {"role": "user", "parts": [
                {"text": f"Answer the user's question about this image clearly and concisely:\n{question}"},
                {"inline_data": {"mime_type": mime, "data": img_bytes}}
            ]}
        ]
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
        answer = getattr(resp, "text", "") or ""
        if not answer:
            for cand in getattr(resp, "candidates", []):
                content = getattr(cand, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        if getattr(part, "text", None):
                            answer += part.text
        if not answer:
            raise RuntimeError("Empty response from Gemini Vision.")
        return answer
    except Exception as e:
        show_error(f"Image Q&A error: {e}")
        raise

# =============================================================================
# Text-to-Image
# =============================================================================
def gemini_generate_image(prompt: str, num_images: int = 1) -> List[bytes]:
    try:
        if not prompt.strip():
            raise RuntimeError("Text-to-image prompt cannot be empty.")
        api_key = st.session_state["gemini_api_key"] or st.secrets.get("GEMINI_KEY") or st.secrets.get("GEMINI_API_KEY")
        api_key = _require_key(api_key, "Gemini API key")
        client = genai.Client(api_key=api_key)
        # Try Imagen 4 first
        try:
            resp = client.models.generate_images(
                model="imagen-4.0-generate-001",
                prompt=prompt,
                config=genai_types.GenerateImagesConfig(number_of_images=max(1, min(num_images, 4)))
            )
            out_images: List[bytes] = []
            for gi in getattr(resp, "generated_images", []):
                image_obj = getattr(gi, "image", None)
                if image_obj and getattr(image_obj, "image_bytes", None):
                    out_images.append(image_obj.image_bytes)
            if out_images:
                return out_images
        except Exception:
            pass  # fallback
        # Fallback to Gemini image-preview
        resp2 = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[prompt]
        )
        out_images2: List[bytes] = []
        try:
            for cand in getattr(resp2, "candidates", []):
                content = getattr(cand, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            out_images2.append(inline.data)
        except Exception:
            pass
        if not out_images2:
            maybe_text = getattr(resp2, "text", "") or ""
            if maybe_text:
                raise RuntimeError(f"Image generation returned no image. Model said: {maybe_text}")
            raise RuntimeError("Image generation returned no image data.")
        return out_images2
    except Exception as e:
        show_error(f"Text-to-Image error: {e}")
        raise

# =============================================================================
# Web Search (Tavily) + Query Checker
# =============================================================================
def needs_web_search(q: str) -> bool:
    ql = (q or "").lower().strip()
    if not ql:
        return False
    triggers = ["today", "latest", "news", "search", "price of", "vs", "compare", "update", "breaking"]
    if any(t in ql for t in triggers):
        return True
    if "http://" in ql or "https://" in ql:
        return True
    last = ql.split()[-1] if ql.split() else ""
    if "." in last and len(last) >= 3:
        return True
    import re
    years = re.findall(r"\b(20[2-9][0-9])\b", ql)
    if any(int(y) >= 2023 for y in years):
        return True
    return False

def tavily_search_snippets(query: str, max_results: int = 5) -> str:
    api_key = st.session_state.get("tavily_api_key") or st.secrets.get("TAVILY_API_KEY")
    if not api_key:
        return ""
    try:
        client = TavilyClient(api_key=api_key)
        res = client.search(
            query=query,
            search_depth="advanced",
            max_results=max(1, min(max_results, 10)),
            include_answer=False,
            include_raw_content=False,
            topic="general",
        )
        items = res.get("results", []) if isinstance(res, dict) else []
        lines = []
        for i, it in enumerate(items[:max_results], start=1):
            title = it.get("title") or "Result"
            url = it.get("url") or ""
            content = (it.get("content") or "").strip().replace("\n", " ")
            if len(content) > 300:
                content = content[:300] + "…"
            lines.append(f"[{i}] {title} — {content} ({url})")
        return "\n".join(lines)
    except Exception as e:
        show_warn(f"Tavily search failed: {e}")
        return ""

def augment_with_web(prompt_text: str, web_snippets: str) -> str:
    if not web_snippets:
        return prompt_text
    if st.session_state["selected_model"].startswith("OpenAI"):
        try:
            payload = json.loads(prompt_text)
            payload["messages"].append({"role": "system", "content": "Relevant web context:\n" + web_snippets})
            return json.dumps(payload)
        except Exception:
            pass
    return f"Relevant web context:\n{web_snippets}\n\n{prompt_text}"

# =============================================================================
# Layout: Header
# =============================================================================
st.title("Multimodal Chatbot")
st.caption("Two-pane workspace: chat + media on the left, Text‑to‑Image studio on the right.")

# =============================================================================
# Two-Pane Main Layout (after sidebar)
# =============================================================================
left_pane, right_pane = st.columns([1, 1], vertical_alignment="top")

# =======================
# LEFT PANE (Chat + Media)
# =======================
with left_pane:
    st.markdown('<div class="left-pane">', unsafe_allow_html=True)

    # 1) Chat transcript at the very top (messages render here)
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 2) Chat input sits ABOVE the media section
    user_input = st.chat_input("Enter your message...")

    # 3) Compact media panel (below input): image uploader and voice record
    with st.container(border=True):
        st.markdown('<div class="media-card media-narrow">', unsafe_allow_html=True)
        st.subheader("Media")

        image_uploaded = st.file_uploader(
            "Upload image (optional)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            help="Upload an image for analysis or reference."
        )
        if image_uploaded:
            try:
                img = Image.open(image_uploaded)
                st.image(img, use_container_width=True, clamp=True)
                image_uploaded.seek(0)
                original_bytes = image_uploaded.read()
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.download_button(
                        "Download",
                        data=original_bytes,
                        file_name=image_uploaded.name,
                        mime=image_uploaded.type or "image/png"
                    )
                with c2:
                    st.caption(f"{image_uploaded.size/1024:.0f} KB")
                image_uploaded.seek(0)
            except Exception as e:
                show_warn(f"Preview error: {e}")

        audio_value = st.audio_input("Voice (optional)")
        if audio_value is not None:
            st.audio(audio_value, format="audio/wav")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# RIGHT PANE (Dedicated TTI Studio Card)
# =====================================
with right_pane:
    st.markdown('<div class="tii-card">', unsafe_allow_html=True)
    st.markdown("<h3>🎨 Text to Image Studio</h3>", unsafe_allow_html=True)
    st.markdown('<div class="tii-subtle">Design compelling visuals. Add style hints (cinematic, watercolor, low‑poly) and aspect ratios.</div>', unsafe_allow_html=True)

    tii_prompt = st.text_area(
        "Prompt",
        value=st.session_state.get("tii_prompt", ""),
        height=110,
        placeholder="e.g., A cozy reading nook with warm lighting and plants, cinematic, 4k"
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        num_imgs = st.slider("Count", min_value=1, max_value=4, value=1, help="Number of variations")
    with c2:
        style_hint = st.selectbox("Style", ["None", "Cinematic", "Digital Art", "Photoreal", "Watercolor", "Low-poly"], index=0)
    with c3:
        aspect = st.selectbox("Aspect", ["Auto", "1:1", "3:2", "16:9", "9:16"], index=0)

    gen_btn = st.button("Generate Images", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# STT-only → seed prompt if user spoke without typing
# =============================================================================
prompt_seed = None
if 'audio_value' in locals() and audio_value and not (user_input and user_input.strip()):
    st.info(f"Transcribing with {st.session_state['stt_backend']} ...")
    audio_bytes = audio_value.getvalue()
    transcript = stt_openai(audio_bytes) if st.session_state["stt_backend"] == "OpenAI Whisper" else stt_gemini(audio_bytes)
    if transcript:
        prompt_seed = f"[Voice Transcript]\n{transcript}"

# =============================================================================
# TTI Generation (Right Pane) — minimal loader
# =============================================================================
if gen_btn:
    def do_generate():
        full_prompt = tii_prompt
        if style_hint and style_hint != "None":
            full_prompt += f"\nStyle: {style_hint}"
        if aspect and aspect != "Auto":
            full_prompt += f"\nAspect: {aspect}"
        imgs = gemini_generate_image(full_prompt, num_images=num_imgs)
        for _b in imgs:
            st.session_state["generated_images"].append({"bytes": _b, "name": f"generated_{len(st.session_state['generated_images'])+1}.png"})
        return f"ok:{len(imgs)}"

    try:
        st.session_state["tii_prompt"] = tii_prompt
        tti_placeholder = st.empty()
        with tti_placeholder.container():
            st.markdown('<div class="loader-wrap">', unsafe_allow_html=True)
            tti_msg = st.empty()
            tti_msg.info(minimal_loader_text("generating"))
            _ = do_generate()
            st.markdown('</div>', unsafe_allow_html=True)
        tti_placeholder.success("Image(s) generated.")
    except Exception as e:
        show_error(str(e))

# Render generated gallery beneath the studio
if st.session_state["generated_images"]:
    st.subheader("Generated Images")
    gcols = st.columns(3)
    for i, item in enumerate(st.session_state["generated_images"]):
        with gcols[i % 3]:
            try:
                img = Image.open(io.BytesIO(item["bytes"]))
                st.image(img, use_container_width=True)
                st.download_button(
                    "Download",
                    data=item["bytes"],
                    file_name=item["name"],
                    mime="image/png",
                    key=f"dl_{i}_{item['name']}"
                )
            except Exception as e:
                show_warn(f"Display error: {e}")

# =============================================================================
# Chat Handling (Left Pane) — loader replaced in-place by assistant message
# =============================================================================
def route_and_shape(prompt_text: str) -> str:
    try:
        final_input = prompt_text
        if st.session_state["enable_router"]:
            system_msg, user_msg = route_and_polish(prompt_text)
            if st.session_state["selected_model"].startswith("OpenAI"):
                final_input = json.dumps({
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ]
                })
            elif st.session_state["selected_model"].startswith("groq"):
                final_input = f"[System]: {system_msg}\n[User]: {user_msg}"
            else:
                final_input = f"{system_msg}\n\nUser: {user_msg}"
        return final_input
    except Exception as e:
        show_warn(f"Router warning: {e}")
        return prompt_text

prompt = user_input or prompt_seed
if prompt:
    # Append user bubble to top transcript
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Decide if vision answer or text model
    use_image_qa = st.session_state.get("image_qa_mode", False) and ('image_uploaded' in locals() and image_uploaded is not None)
    final_input = route_and_shape(prompt)

    # Web router — only when enabled, not doing image QA
    web_snips = ""
    st.session_state["web_used"] = False
    if st.session_state.get("enable_web_router", True) and not use_image_qa and needs_web_search(prompt):
        web_snips = tavily_search_snippets(prompt, max_results=5)
        if web_snips:
            st.session_state["web_used"] = True
            final_input = augment_with_web(final_input, web_snips)

    def do_infer():
        if use_image_qa:
            return gemini_vision_answer(prompt, image_uploaded)
        return run_model(st.session_state["selected_model"], final_input)

    # Create an assistant bubble placeholder and REPLACE it with the final response
    with chat_container:
        with st.chat_message("assistant"):
            assistant_placeholder = st.empty()
            try:
                loader_kind = "loading" if use_image_qa else "thinking"
                response_text = rotating_spinner_content(assistant_placeholder, do_infer, kind=loader_kind)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                assistant_placeholder.error(f"Request failed: {e}")
                st.info("Try: set API key(s) via st.secrets or sidebar overrides, switch model, verify access/region, or re-run the request.")

# bee_app_cv_internal_only.py
# OpenCV drives final_behavior internally (no CV diagnostics shown in UI)
# Base Apollo: general description
# Finetuned Apollo: bee-specific analysis conditioned on CV label + Stage-1 text
# Optional Perceiver (best-effort) + requested Gradio theme

import os
os.environ["GRADIO_USE_BROTLI"] = "0"

import sys, asyncio, time, traceback, threading, re
from statistics import median

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, TextIteratorStreamer

from utils.mm_utils import KeywordsStoppingCriteria, tokenizer_mm_token, ApolloMMLoader
from utils.conversation import conv_templates, SeparatorStyle
from utils.constants import X_START_TOKEN, X_END_TOKEN

# ---- OpenCV ----
import cv2
import numpy as np

# -----------------------------
# Config
# -----------------------------
MODEL_REPO = "GoodiesHere/Apollo-LMMs-Apollo-3B-t32"
CKPT_PATH  = "checkpoints/apollo_bee_vqa_best_cpu.pth"
PORT = 7860
BYTES_IN_MB = 1024 * 1024

# Windows event loop tweak
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


# =========================================================
# Perceiver Resampler (best-effort)
# =========================================================
import torch.nn as nn

class PerceiverResampler(nn.Module):
    """
    Perceiver-style resampler:
    reduces N vision tokens -> M latent tokens using cross-attention.
    Only applied if the tensor is [B, N, D].
    """
    def __init__(self, dim, num_latents=64, num_heads=8, depth=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) / (dim ** 0.5))
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
                activation="gelu", batch_first=True
            )
            for _ in range(depth)
        ])

    def forward(self, vision_tokens):  # [B, N, D]
        B, N, D = vision_tokens.shape
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]
        for blk in self.blocks:
            latents = blk(tgt=latents, memory=vision_tokens)
        return latents


# =========================================================
# Helpers
# =========================================================
def _normalize_path(p):
    if isinstance(p, dict):
        return p.get("name") or p.get("path")
    return p

def _align_mm_dtype(x, dtype, device):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_align_mm_dtype(t, dtype, device) for t in x)
    if isinstance(x, dict):
        return {k: _align_mm_dtype(v, dtype, device) for k, v in x.items()}
    return x

def safe_mm_tokenize(prompt_text: str, tokenizer, device):
    """
    Fix for Apollo crash:
    tokenizer_mm_token(...) can return None in some cases.
    This fallback guarantees input_ids is always a tensor.
    """
    try:
        ids = tokenizer_mm_token(prompt_text, tokenizer, return_tensors="pt")
        if ids is None:
            raise ValueError("tokenizer_mm_token returned None")
        ids = ids.unsqueeze(0)
    except Exception:
        ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids
    return ids.to(device)

def wait_for_complete_file(
    path: str,
    *,
    min_size: int = 1024,
    stable_checks: int = 6,
    interval: float = 0.3,
    timeout: float = 300.0,
    progress: gr.Progress | None = None,
) -> bool:
    """
    Safe upload completion check.
    IMPORTANT: progress can be None. We never call it if None.
    """
    start = time.time()
    last = -1
    stable = 0

    if progress:
        progress(0.02, desc="Receiving…", total=1)

    while time.time() - start < timeout:
        size = 0
        if os.path.exists(path):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0

        if progress:
            elapsed = time.time() - start
            frac = min(0.9, max(0.02, elapsed / timeout))
            progress(frac, desc=f"Receiving… {size / BYTES_IN_MB:.1f} MB", total=1)

        if size >= min_size:
            if size == last:
                stable += 1
                if stable >= stable_checks:
                    if progress:
                        progress(0.95, desc="Upload complete.", total=1)
                    return True
            else:
                stable = 0
            last = size

        time.sleep(interval)

    return False


# =========================================================
# Deterministic summary override
# =========================================================
def write_in_summary(final_behavior: str, cv_res: dict | None, general_text: str) -> str:
    b = (final_behavior or "unknown").lower()
    wing = cv_res.get("wing_motion_mean") if cv_res else None

    if b == "signaling":
        return (
            "The bees are signaling/communicating. Multiple bees gather closely and repeatedly interact at short range, "
            "which matches communication-style behavior rather than ventilation or self-cleaning."
        )

    if b == "fanning":
        extra_line = f" (wing score ≈ {wing})" if wing is not None else ""
        return (
            "The bees are fanning. This is characterized by sustained, rapid wing beating that produces airflow for hive "
            "ventilation and moisture/temperature regulation."
            + extra_line
        )

    if b == "grooming":
        return (
            "The bees are grooming. Grooming involves leg and mouthpart motions used to clean the body and antennae, "
            "and it lacks sustained fanning-style wing beating."
        )

    return "The behavior cannot be determined confidently from the available visual cues."

def enforce_behavior_consistency(full_text: str, final_behavior: str) -> str:
    if not full_text:
        return full_text
    b = (final_behavior or "unknown").lower()
    out = full_text

    if b == "signaling":
        out = re.sub(r"\bfann\w*\b", "signal", out, flags=re.IGNORECASE)
        out = re.sub(r"\bventilat\w*\b", "communicat", out, flags=re.IGNORECASE)
    if b == "grooming":
        out = re.sub(r"\bfann\w*\b", "groom", out, flags=re.IGNORECASE)
    if b == "fanning" and "fanning" not in out.lower():
        out += "\n\nThis behavior is best classified as fanning due to sustained wing motion."
    return out


# =========================================================
# OpenCV behavior detector (minimal proxy)
# Replace with your blackhat+motion ROI version if you want.
# =========================================================
MAX_FRAMES = 300
FRAME_STEP = 3
CLUSTER_MIN_BEES = 3
PRESENCE_RATIO_THR = 0.30
WING_MOTION_THR_DOWN = 0.00234
WING_MOTION_THR_UP   = 0.00408

def cv_analyze(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "reason": f"could_not_open: {video_path}"}

    bees_per_frame = []
    wing_scores = []
    prev_gray = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % FRAME_STEP != 0:
                continue
            if frame_idx > MAX_FRAMES:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # crude bee proxy
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([])
            bees_count = int(np.sum((areas > 200) & (areas < 20000)))
            bees_per_frame.append(bees_count)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                wing_scores.append(float(diff.mean() / 255.0))
            prev_gray = gray
    finally:
        cap.release()

    bees_med = int(median(bees_per_frame)) if bees_per_frame else 0
    wing_mean = float(np.mean(wing_scores)) if wing_scores else 0.0
    presence_ratio = float(np.mean([b >= CLUSTER_MIN_BEES for b in bees_per_frame])) if bees_per_frame else 0.0

    clustered = (bees_med >= CLUSTER_MIN_BEES) and (presence_ratio >= PRESENCE_RATIO_THR)

    if clustered:
        behavior = "signaling"
    else:
        behavior = "fanning" if (WING_MOTION_THR_DOWN <= wing_mean <= WING_MOTION_THR_UP) else "grooming"

    return {
        "ok": True,
        "bees_median": bees_med,
        "presence_ratio": round(presence_ratio, 3),
        "wing_motion_mean": round(wing_mean, 5),
        "clustered": bool(clustered),
        "behavior": behavior,
        "frames_used": len(bees_per_frame),
    }


# =========================================================
# Apollo wrapper (streaming)
# =========================================================
class ApolloVideoChat:
    def __init__(self, repo_id: str, ckpt_path: str | None, mode: str, use_perceiver: bool = False):
        self.mode = mode
        self.use_perceiver = use_perceiver

        model_path = snapshot_download(repo_id, repo_type="model")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        attn_impl = "sdpa" if torch.__version__ > "2.1.2" else "eager"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        ).to(device=device, dtype=dtype)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[{mode}] Loading finetuned weights: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            sd = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(sd, strict=False)
        else:
            if ckpt_path:
                print(f"[{mode}] WARNING: checkpoint not found: {ckpt_path} (using base)")

        for attr in ["vision_tower", "mm_connector"]:
            try:
                getattr(model, attr).to(device=device, dtype=dtype)
            except Exception:
                pass

        model.eval()

        self.model = model
        self.tokenizer = model.tokenizer
        self.vision_processors = model.vision_tower.vision_processor
        self.cfg = model.config
        self.version = "qwen_1_5"
        self.mm_use_im_start_end = self.cfg.use_mm_start_end
        self.mm_loader = None

        hidden = getattr(self.cfg, "hidden_size", 3072)
        self.resampler = PerceiverResampler(dim=hidden, num_latents=64, num_heads=8, depth=2).to(
            device=device, dtype=dtype
        )

    def attach_loader(self, loader: ApolloMMLoader):
        self.mm_loader = loader

    def _apply_first_prompt(self, message: str, media_prompt: str, data_type: str) -> str:
        if (X_START_TOKEN[data_type] in media_prompt) and (X_END_TOKEN[data_type] in media_prompt):
            pre = media_prompt
        elif self.mm_use_im_start_end:
            pre = X_START_TOKEN[data_type] + media_prompt + X_END_TOKEN[data_type]
        else:
            pre = media_prompt
        return pre + "\n\n" + message

    def _build_instruction(self, user_question: str, behavior_hint: str | None = None, general_description: str | None = None) -> str:
        if self.mode == "general":
            return (
                "You are an expert observer of honey bee videos.\n"
                "Describe in rich detail what happens in this video.\n"
                "Focus on visible objects, motions, interactions, and scene changes.\n"
                "Use 4–8 full sentences and stay grounded ONLY in what is clearly visible.\n"
                "If you cannot see something clearly, say that it is unclear instead of guessing.\n"
                f"Question: {user_question.strip()}"
            )

        behavior_line = ""
        if behavior_hint == "signaling":
            behavior_line = "The main behavior is signaling/communication among multiple bees gathered closely.\n"
        elif behavior_hint == "fanning":
            behavior_line = "The main behavior is fanning with sustained rapid wing beating.\n"
        elif behavior_hint == "grooming":
            behavior_line = "The main behavior is grooming with legs and mouthparts cleaning body/antennae.\n"

        base_desc_line = ""
        if general_description:
            base_desc_line = (
                "Base visual description (treat as ground truth):\n"
                f"\"{general_description.strip()}\"\n"
                "Do not contradict it.\n"
            )

        return (
            "You are an expert in honey bee behavior.\n"
            + base_desc_line
            + behavior_line
            + "Provide a bee-specific analysis grounded in visible motion.\n"
            f"Question: {user_question.strip()}"
        )

    def _perceiver_reduce(self, video_tensor):
        if not self.use_perceiver:
            return video_tensor
        try:
            if isinstance(video_tensor, torch.Tensor) and video_tensor.dim() == 3:
                return self.resampler(video_tensor)
        except Exception:
            pass
        return video_tensor

    def stream_answer_about_video(
        self,
        video_tensor,
        vision_prompt: str,
        user_question: str,
        temperature=0.25,
        top_p=0.8,
        max_new_tokens=256,
        behavior_hint: str | None = None,
        general_description: str | None = None,
    ):
        instruction = self._build_instruction(user_question, behavior_hint, general_description)

        video_tensor = _align_mm_dtype(video_tensor, self.model.dtype, self.model.device)
        video_tensor = self._perceiver_reduce(video_tensor)

        conv = conv_templates[self.version].copy()
        prompt_with_video = self._apply_first_prompt(instruction, vision_prompt, "video")
        conv.append_message(conv.roles[0], prompt_with_video)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = safe_mm_tokenize(prompt_text, self.tokenizer, self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            inputs=input_ids,
            vision_input=[video_tensor],
            data_types=["video"],
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            num_beams=1,
            use_cache=True,
            stopping_criteria=[stopping],
            streamer=streamer,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        return streamer


# =========================================================
# Load models + shared loader
# =========================================================
print(">> Loading base Apollo (general)…")
base_handler = ApolloVideoChat(MODEL_REPO, ckpt_path=None, mode="general", use_perceiver=False)

print(">> Loading finetuned Apollo (bee)…")
bee_handler = ApolloVideoChat(MODEL_REPO, ckpt_path=CKPT_PATH, mode="bee", use_perceiver=True)

device = bee_handler.model.device
frames_per_clip = 4
clip_duration = getattr(bee_handler.cfg, "clip_duration", None)

shared_loader = ApolloMMLoader(
    bee_handler.vision_processors,
    clip_duration,
    frames_per_clip,
    clip_sampling_ratio=0.1,
    model_max_length=bee_handler.cfg.model_max_length,
    device=device,
    num_repeat_token=bee_handler.cfg.mm_connector_cfg["num_output_tokens"],
)

# fps-based knobs (best-effort)
for k, v in {"sampling_mode": "fps", "target_fps": 12, "max_num_clips": 8, "short_side": 336}.items():
    try:
        setattr(shared_loader, k, v)
    except Exception:
        pass

base_handler.attach_loader(shared_loader)
bee_handler.attach_loader(shared_loader)


# =========================================================
# Upload handler
# =========================================================
def on_file_selected(file_input, progress: gr.Progress = gr.Progress(track_tqdm=True)):
    path = _normalize_path(file_input)
    if not path:
        return (gr.update(value=""), gr.update(visible=False), None, "", "")

    if not path.lower().endswith(".mp4"):
        return (gr.update(value="❗ Please upload an MP4 file."), gr.update(visible=True), None, "", "")

    ok = wait_for_complete_file(path, progress=progress)
    if not ok:
        return (gr.update(value="❗ Upload incomplete. Please try again."), gr.update(visible=True), None, "", "")

    progress(0.96, desc="Parsing video…", total=1)
    try:
        video_tensor, vision_prompt = shared_loader.load_video(path)
        video_tensor = _align_mm_dtype(video_tensor, bee_handler.model.dtype, bee_handler.model.device)
    except Exception:
        traceback.print_exc()
        return (gr.update(value="❗ Failed to parse the video. Try another MP4."), gr.update(visible=True), None, "", "")

    size_mb = os.path.getsize(path) / BYTES_IN_MB if os.path.exists(path) else 0
    return (
        gr.update(value=f"✅ Video uploaded ({size_mb:.1f} MB). You can now ask a question →"),
        gr.update(visible=True),
        video_tensor,
        vision_prompt,
        path,
    )


# =========================================================
# Chat (OpenCV -> Stage1 -> Stage2 -> summary)
# CV runs, but CV diagnostics are NOT shown in UI.
# =========================================================
def chat(file_input, question, temperature, top_p, max_output_tokens,
         cached_video, cached_prompt, cached_path,
         progress: gr.Progress = gr.Progress(track_tqdm=True)):

    path = _normalize_path(file_input) or cached_path
    if not question or not question.strip():
        raise gr.Error("Please type your question about the video.")

    # reload if new file is selected, else reuse cached
    if path and (file_input is not None):
        ok = wait_for_complete_file(path, progress=None)
        if not ok:
            raise gr.Error("The upload did not complete. Please re-upload.")
        try:
            video_tensor, vision_prompt = shared_loader.load_video(path)
            video_tensor = _align_mm_dtype(video_tensor, bee_handler.model.dtype, bee_handler.model.device)
        except Exception:
            traceback.print_exc()
            raise gr.Error("Failed to parse the uploaded video. Try another MP4.")
        cached_video, cached_prompt, cached_path = video_tensor, vision_prompt, path
    else:
        if cached_video is None or not cached_prompt:
            raise gr.Error("Please upload an MP4 file first.")
        video_tensor, vision_prompt, path = cached_video, cached_prompt, cached_path

    # --- OpenCV (internal only) ---
    try:
        cv_res = cv_analyze(path)
        final_behavior = cv_res["behavior"] if cv_res.get("ok") else "unknown"
    except Exception:
        cv_res = {"ok": False}
        final_behavior = "unknown"

    cv_txt = ""  # <-- do not show any CV diagnostics

    # Stage-1
    yield (gr.update(), gr.update(value="🧠 Stage 1: describing…", visible=True), gr.update(visible=True),
           gr.update(value=cv_txt), cached_video, cached_prompt, cached_path)

    general_stream = base_handler.stream_answer_about_video(
        video_tensor, vision_prompt, question.strip(),
        temperature=float(temperature), top_p=float(top_p),
        max_new_tokens=int(max_output_tokens) // 2 if max_output_tokens else 192,
    )

    general_text = ""
    for piece in general_stream:
        general_text += piece
        yield (gr.update(), gr.update(value="🧠 Stage 1: describing…", visible=True), gr.update(visible=True),
               gr.update(value="General description:\n" + general_text),
               cached_video, cached_prompt, cached_path)

    # Stage-2 (finetuned, conditioned on CV label + stage-1 description)
    yield (gr.update(), gr.update(value="🧠 Stage 2: finetuned analysis…", visible=True), gr.update(visible=True),
           gr.update(value="General description:\n" + general_text),
           cached_video, cached_prompt, cached_path)

    bee_stream = bee_handler.stream_answer_about_video(
        video_tensor, vision_prompt, question.strip(),
        temperature=float(temperature), top_p=float(top_p),
        max_new_tokens=int(max_output_tokens) if max_output_tokens else 256,
        behavior_hint=final_behavior,
        general_description=general_text,
    )

    bee_analysis = ""
    for piece in bee_stream:
        bee_analysis += piece
        yield (gr.update(), gr.update(value="🧠 Stage 2: generating…", visible=True), gr.update(visible=True),
               gr.update(value="General description:\n" + general_text
                         + "\n\nBee-specific analysis:\n" + bee_analysis),
               cached_video, cached_prompt, cached_path)

    summary_text = write_in_summary(final_behavior, cv_res if cv_res.get("ok") else None, general_text)

    final_answer = (
        "General description:\n" + general_text
        + "\n\nBee-specific analysis:\n" + bee_analysis
        + "\n\nIn summary,\n" + summary_text
    )
    final_answer = enforce_behavior_consistency(final_answer, final_behavior)

    yield (gr.update(), gr.update(value="✅ Done.", visible=True), gr.update(visible=True),
           gr.update(value=final_answer), cached_video, cached_prompt, cached_path)


# =========================================================
# UI
# =========================================================
with gr.Blocks(
    title="Bee Activity QA — Apollo-3B (OpenCV internal + finetuned + Perceiver)",
    theme=gr.themes.Default(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky),
) as demo:
    gr.Markdown(
        """
        <div style="text-align:center">
          <h2>Bee Activity QA — Feel Free to ask any questions</h2>
        </div>
        """
    )

    cached_video = gr.State(None)
    cached_prompt = gr.State("")
    cached_path = gr.State("")

    with gr.Row():
        with gr.Column(scale=4):
            video_file = gr.File(label="Upload Video (MP4)", file_types=[".mp4"], file_count="single")
            upload_status = gr.Markdown(visible=False)

            with gr.Accordion("Generation parameters", open=False):
                temperature = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Top P")
                max_tokens = gr.Slider(128, 512, value=320, step=32, label="Max new tokens")

        with gr.Column(scale=8):
            question = gr.Textbox(label="Your question", placeholder="e.g., Describe the video?")
            send = gr.Button("Send", variant="primary")
            answer = gr.Textbox(label="Answer", lines=18)

    video_file.change(
        on_file_selected,
        inputs=[video_file],
        outputs=[upload_status, upload_status, cached_video, cached_prompt, cached_path],
        queue=True,
    )

    send.click(
        chat,
        [video_file, question, temperature, top_p, max_tokens, cached_video, cached_prompt, cached_path],
        [video_file, upload_status, upload_status, answer, cached_video, cached_prompt, cached_path],
    )
    question.submit(
        chat,
        [video_file, question, temperature, top_p, max_tokens, cached_video, cached_prompt, cached_path],
        [video_file, upload_status, upload_status, answer, cached_video, cached_prompt, cached_path],
    )

if __name__ == "__main__":
    demo.queue(max_size=32, default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=PORT, show_error=True, inbrowser=False)
    print(f"\n👉 App is running at: http://127.0.0.1:{PORT}\n")

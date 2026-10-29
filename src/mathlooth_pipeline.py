"""
Mathlooth (مثلوث): Arabic Text-to-2D-to-3D Saudi Cultural Object Generation
==========================================================================

Inference pipeline deployed on Modal cloud GPUs:

    1. Prompt rewriting  : Qwen3-8B guided by a Saudi heritage knowledge graph
    2. Text-to-2D        : Stable Diffusion XL + fine-tuned LoRA
    3. 2D-to-3D          : TripoSG + LoRA fine-tuned with Cultural Alignment Loss (CAL)

Graduation Project, College of Computer and Information Sciences,
Princess Nourah bint Abdulrahman University, Riyadh.

Usage:
    modal run src/mathlooth_pipeline.py --prompt "دلة قهوة سعودية بلون ذهبي"
"""

import modal

# ---------------------------------------------------------------------------
# Modal app, container image and volumes
# ---------------------------------------------------------------------------
# Expected local layout (not included in this repository):
#   models/knowledge_graph.json        Saudi heritage knowledge graph
#   models/sd_lora/                    fine-tuned SDXL LoRA weights
#   models/triposg_lora/               fine-tuned TripoSG LoRA weights (PEFT)

app = modal.App("mathlooth")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "transformers", "diffusers", "accelerate", "peft",
        "safetensors", "huggingface_hub", "rembg", "onnxruntime",
        "pymcubes", "trimesh", "pillow", "numpy",
    )
    .run_commands("git clone https://github.com/VAST-AI-Research/TripoSG /opt/TripoSG")
    .env({"PYTHONPATH": "/opt/TripoSG"})
    .add_local_dir("models", remote_path="/models")
)

hf_volume = modal.Volume.from_name("mathlooth-hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# [1] Knowledge Graph — Prompt Detection & Fact Extraction
# ---------------------------------------------------------------------------
# KG rewriting helpers — adapted to knowledge_graph.json structure:
#   { "objects": { key: {name_en, name_ar, category, materials, form,
#                        functional_parts, ornaments, cultural_context,
#                        cultural_identity, colors} },
#     "aliases": { english_phrase: object_key } }
# ---------------------------------------------------------------------------

def _safe_join(values, limit=None):
    if not values:
        return ""
    vals = [str(v).strip() for v in values if str(v).strip()]
    if limit is not None:
        vals = vals[:limit]
    return ", ".join(vals)


def _load_kg(path: str):
    import json
    with open(path, "r", encoding="utf-8") as f:
        kg = json.load(f)
    return kg.get("objects", {}), kg.get("aliases", {})


def _detect_kg_key(user_prompt: str, objects: dict, aliases: dict) -> str | None:
    """
    Return the object key from knowledge_graph.json that best matches the
    user's prompt, or None if no match.

    Priority: aliases dict (longer phrases win) → direct object key match.
    The aliases section in the KG already covers all common English phrasings.
    """
    prompt_lower = user_prompt.lower()
    for alias in sorted(aliases, key=len, reverse=True):
        if alias.lower() in prompt_lower:
            return aliases[alias]
    for key in objects:
        if key.lower() in prompt_lower:
            return key
    return None


def _empty_fact_block():
    return {
        "object_name": "", "subtype": "", "category": "",
        "materials": [], "primary_shapes": [], "symmetry": "",
        "silhouette": "", "surface_finish": [], "common_marks": [],
        "must_have": [], "avoid": [], "proportions": [], "parts": [],
    }


def _build_kg_fact_block(obj_key: str, obj_data: dict) -> dict:
    """Map knowledge_graph.json fields → fact_block expected by _build_qwen_instruction."""
    return {
        "object_name": obj_data.get("name_en", obj_key),
        "subtype": obj_data.get("name_ar", obj_key),
        "category": obj_data.get("category", "heritage object"),
        "materials": obj_data.get("materials", []),
        # form is a free-text shape description — wrap as single-item list
        "primary_shapes": [obj_data["form"]] if obj_data.get("form") else [],
        "symmetry": "",
        "silhouette": obj_data.get("form", ""),
        "surface_finish": obj_data.get("colors", []),
        "common_marks": obj_data.get("ornaments", []),
        # functional_parts are the visually essential components → must_have
        "must_have": obj_data.get("functional_parts", []),
        "avoid": [],
        "proportions": [],
        "parts": obj_data.get("functional_parts", []),
    }


def _get_visual_anchor(obj_key: str, obj_data: dict) -> str:
    """Use name_en from KG as the visual anchor phrase."""
    return obj_data.get("name_en", "museum heritage object")


def _build_qwen_instruction(raw_caption: str, kg_key: str, fact_block: dict, anchor: str = "") -> str:
    if not anchor:
        anchor = "museum heritage object"
    fb = fact_block
    return f"""You are rewriting heritage object captions for SDXL image generation.

Task:
Given a raw caption, object identity from a knowledge graph, structured heritage facts, and a visual anchor phrase, rewrite into one concise visual prompt and one concise negative prompt.

Requirements:
- preserve Saudi cultural identity and object structure
- keep it SINGLE OBJECT only, do NOT describe a scene
- prefer realistic product-photo or museum-catalog style
- always bias toward isolated artifact photography
- include must-have features in visual_prompt; put avoid features only in negative_prompt
- include style cues: museum catalog photo, isolated object, studio lighting, high detail, neutral background
- include the visual anchor naturally in visual_prompt
- output valid JSON only, no markdown fences

Return exactly:
{{"visual_prompt": "string", "negative_prompt": "string"}}

Important:
- both prompts must be in English and must not be empty
- negative_prompt must suppress: people, hands, scenes, room interiors, multiple objects, clutter, modern variants

kg_key: {kg_key}
visual_anchor: {anchor}

Raw caption:
{raw_caption}

Knowledge facts:
object_name: {fb["object_name"]}
subtype: {fb["subtype"]}
category: {fb["category"]}
materials: {_safe_join(fb["materials"], 6)}
primary_shapes: {_safe_join(fb["primary_shapes"], 6)}
symmetry: {fb["symmetry"]}
silhouette: {fb["silhouette"]}
surface_finish: {_safe_join(fb["surface_finish"], 5)}
common_marks: {_safe_join(fb["common_marks"], 6)}
must_have: {_safe_join(fb["must_have"], 6)}
avoid: {_safe_join(fb["avoid"], 6)}
proportions: {_safe_join(fb["proportions"], 6)}
parts: {_safe_join(fb["parts"], 8)}

Return JSON only."""


# ---------------------------------------------------------------------------
# [2] Stage 1 — Qwen3-8B KG-Guided Prompt Rewriting
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=900,
)
def rewrite_prompt(user_prompt: str) -> dict:
    """
    Rewrite the user's prompt into {visual_prompt, negative_prompt} using
    Qwen3-8B + KG fact lookup.
    """
    import json
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # --- KG lookup ---
    objects, aliases = _load_kg("/models/knowledge_graph.json")
    kg_key = _detect_kg_key(user_prompt, objects, aliases)

    if kg_key and kg_key in objects:
        obj_data = objects[kg_key]
        fact_block = _build_kg_fact_block(kg_key, obj_data)
        anchor = _get_visual_anchor(kg_key, obj_data)
    else:
        fact_block = _empty_fact_block()
        anchor = "museum heritage object"
        kg_key = kg_key or ""

    instruction = _build_qwen_instruction(user_prompt, kg_key, fact_block, anchor)

    # --- Qwen3-8B inference ---
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # disable Qwen3 chain-of-thought to get clean JSON output
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # --- Parse JSON output ---
    try:
        parsed = json.loads(generated)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", generated, flags=re.DOTALL)
        parsed = json.loads(m.group(0)) if m else {}

    visual_prompt = str(parsed.get("visual_prompt", user_prompt)).strip()
    negative_prompt = str(parsed.get("negative_prompt", "")).strip()

    if not visual_prompt:
        visual_prompt = user_prompt
    if not negative_prompt:
        negative_prompt = "people, hands, scene, clutter, multiple objects, modern style"

    return {"visual_prompt": visual_prompt, "negative_prompt": negative_prompt}


# ---------------------------------------------------------------------------
# [3] Stage 2 — SDXL + Fine-tuned UNet LoRA → 2D Image
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=900,
)
def generate_2d(prompts: dict) -> bytes:
    """Generate a 2D heritage object image. Returns PNG bytes."""
    import io
    import torch
    from diffusers import StableDiffusionXLPipeline
    from safetensors.torch import load_file

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    # LoRA is in diffusers format (unet.*.lora.down/up.weight keys),
    # so use diffusers' own loader instead of PeftModel.
    lora_sd = load_file("/models/sd_lora/adapter_model.safetensors")
    pipe.load_lora_weights(lora_sd)
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    image = pipe(
        prompt=prompts["visual_prompt"],
        negative_prompt=prompts["negative_prompt"],
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# [4] Stage 3 — TripoSG + EMA/CAL LoRA → 3D Mesh
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=900,
)
def generate_3d(image_bytes: bytes) -> bytes:
    """Generate a GLB 3D mesh from a PNG image. Returns GLB bytes."""
    import io
    import sys
    import torch
    import numpy as np
    import mcubes
    from types import SimpleNamespace
    from PIL import Image
    from peft import PeftModel
    from rembg import remove as rembg_remove

    # Mock diso before TripoSG imports it. diso provides differentiable dual
    # marching cubes (needed for training gradients) but at inference we only
    # need a forward pass. PyMCubes gives identical mesh quality for inference.
    class _MCubesDMC:
        def __init__(self, dtype=torch.float32):
            self.dtype = dtype

        def to(self, device):
            return self

        def __call__(self, sdf, deform=None, return_quads=False, normalize=False, **kwargs):
            import numpy as np
            sdf_np = sdf.squeeze().detach().cpu().float().numpy()
            # NaN marks unqueried octree cells — treat as exterior (large positive)
            sdf_np = np.nan_to_num(sdf_np, nan=1.0)
            verts, faces = mcubes.marching_cubes(-sdf_np, 0)
            verts_t = torch.tensor(verts.copy(), dtype=self.dtype, device=sdf.device)
            faces_t = torch.tensor(faces.astype("int64"), dtype=torch.long, device=sdf.device)
            return verts_t, faces_t

    sys.modules["diso"] = SimpleNamespace(DiffDMC=_MCubesDMC)

    from huggingface_hub import snapshot_download
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from triposg.models.autoencoders import TripoSGVAEModel
    from triposg.models.transformers import TripoSGDiTModel
    from triposg.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler
    from transformers import Dinov2Model, BitImageProcessor

    # TripoSGPipeline.from_pretrained fails because model_index.json references
    # triposg.* module paths that diffusers cannot resolve as HF repo files.
    # Fix: download weights, then load each component directly and construct
    # the pipeline, bypassing diffusers' custom class resolver entirely.
    model_dir = snapshot_download("VAST-AI/TripoSG")

    vae = TripoSGVAEModel.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float16)
    transformer = TripoSGDiTModel.from_pretrained(model_dir, subfolder="transformer", torch_dtype=torch.float16)
    scheduler = RectifiedFlowScheduler.from_pretrained(model_dir, subfolder="scheduler")
    image_encoder = Dinov2Model.from_pretrained(model_dir, subfolder="image_encoder_dinov2", torch_dtype=torch.float16)
    feature_extractor = BitImageProcessor.from_pretrained(model_dir, subfolder="feature_extractor_dinov2")

    pipe = TripoSGPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        image_encoder_dinov2=image_encoder,
        feature_extractor_dinov2=feature_extractor,
    ).to("cuda")

    # Load PEFT LoRA onto transformer then merge+unload so TripoSG's internal
    # calls get a plain transformer with LoRA weights baked in.
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        "/models/triposg_lora/",
    )
    pipe.transformer = pipe.transformer.merge_and_unload()

    # Remove background before 3D generation
    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    input_image = rembg_remove(raw_image).convert("RGB")

    # TripoSG is not text-conditioned; no guidance_scale parameter
    outputs = pipe(image=input_image, num_inference_steps=50)

    # TripoSG output: handle both .mesh and .meshes[0] attribute names
    if hasattr(outputs, "mesh"):
        mesh = outputs.mesh
    elif hasattr(outputs, "meshes"):
        mesh = outputs.meshes[0]
    else:
        mesh = outputs  # pipeline may return the mesh directly

    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# [5] Pipeline Orchestrator
# ---------------------------------------------------------------------------

@app.function(image=image, timeout=3600)
def run_pipeline(user_prompt: str) -> tuple:
    """
    Full pipeline. Returns (visual_prompt: str, image_bytes: bytes, mesh_bytes: bytes).
    Called from the Gradio UI or CLI.
    """
    prompts = rewrite_prompt.remote(user_prompt)
    image_bytes = generate_2d.remote(prompts)
    mesh_bytes = generate_3d.remote(image_bytes)
    return prompts["visual_prompt"], image_bytes, mesh_bytes


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(prompt: str, out_dir: str = "outputs"):
    """Run the full pipeline and save the 2D image and 3D mesh locally."""
    from pathlib import Path

    visual_prompt, image_bytes, mesh_bytes = run_pipeline.remote(prompt)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "object.png").write_bytes(image_bytes)
    (out / "object.glb").write_bytes(mesh_bytes)

    print(f"Enhanced prompt: {visual_prompt}")
    print(f"Saved: {out / 'object.png'} and {out / 'object.glb'}")

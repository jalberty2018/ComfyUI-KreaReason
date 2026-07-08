"""ComfyUI-KreaReason — reason-then-encode for the Krea 2 text encoder (Qwen3-VL-4B).

Krea 2's text encoder is a full Qwen3-VL-4B with a working LM head, so it can rewrite a short
prompt into a rich one — or read a reference image and write the prompt from it — BEFORE
conditioning. One node, running on the exact model that conditions, no second LLM needed.
"""

import torch

CATEGORY = "ShootTheSound/KreaReason"


def _esc(s):
    # the template goes through str.format(text); stray braces would raise or corrupt
    return s.replace("{", "{{").replace("}", "}}")


# Krea 2's trained system descriptor. Overriding the template MUST keep a system role:
# krea2.encode_token_weights strips up to the 2nd <|im_start|> (system, then user), so a
# system-less template would count [user, assistant] and strip into the real prompt.
KREA2_DESCRIPTOR = ("Describe the image by detailing the color, shape, size, texture, "
                    "quantity, text, spatial relationships of the objects and background:")


def build_template(system_prompt=None, reasoning=None):
    if not system_prompt and not reasoning:
        return None  # use the Krea 2 default template (its descriptor system prompt)
    sys = _esc(system_prompt) if system_prompt else KREA2_DESCRIPTOR
    head = "<|im_start|>system\n{}<|im_end|>\n".format(sys)
    body = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    tail = "<think>\n{}\n</think>\n\n".format(_esc(reasoning)) if reasoning else ""
    return head + body + tail


def _encode(clip, text, system_prompt=None, reasoning=None):
    """Tokenize + encode `text` to a Krea 2 CONDITIONING (optionally with a system/think template)."""
    tmpl = build_template((system_prompt or "").strip() or None, (reasoning or "").strip() or None)
    tokens = clip.tokenize(text, **({"llama_template": tmpl} if tmpl else {}))
    return clip.encode_from_tokens_scheduled(tokens)


def _scale_conditioning(cond, k):
    """Multiply the positive conditioning by scalar k, preserving each entry's dict. Krea Turbo is
    CFG-free, so scaling the positive acts like a guidance/strength boost. Effect saturates ~6-8x."""
    if float(k) == 1.0:
        return cond
    k = float(k)
    out = []
    for entry in cond:
        t = entry[0] * k
        d = dict(entry[1]) if len(entry) > 1 and isinstance(entry[1], dict) else (entry[1] if len(entry) > 1 else {})
        if isinstance(d, dict) and "pooled_output" in d and torch.is_tensor(d["pooled_output"]):
            d = {**d, "pooled_output": d["pooled_output"] * k}
        out.append([t, d])
    return out


def _cap_image_tensor(img, megapixels):
    """Downscale a ComfyUI IMAGE (B,H,W,C float 0..1) so H*W <= megapixels*1024*1024 (never upscale)."""
    b, h, w, c = img.shape
    cap = int(float(megapixels) * 1024 * 1024)
    if h * w <= cap or h == 0 or w == 0:
        return img
    s = (cap / (h * w)) ** 0.5
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    x = torch.nn.functional.interpolate(img.movedim(-1, 1).float(), size=(nh, nw),
                                        mode="bilinear", align_corners=False)
    return x.movedim(1, -1)


REASON_INSTRUCTION = (
    "You are an expert prompt engineer for a text-to-image model. Expand the user's prompt into a "
    "single vivid, detailed image-generation prompt. Preserve every subject, action, and relationship "
    "they gave; do not add new characters or objects. Enrich it with concrete lighting, mood, "
    "composition, camera/lens, and visual style. Output ONE flowing paragraph, no preamble, no lists."
)

# --- Reference-image handling: a 3-pass pipeline -------------------------------------------------
# Pass 1 (vision): describe the image in full. Pass 2 (text): filter that description by REMOVING
# the elements image_remove selects. Pass 3 (text): combine the filtered description with the user's
# prompt. Decomposing it this way is far more reliable than a single "describe but ignore X" instruction.

DESCRIBE_FULL_INSTRUCTION = (
    "Describe this image in thorough, concrete, factual detail: the main subject and their "
    "appearance, any clothing, the pose, the background and setting, the lighting, the colors, "
    "the composition and camera angle, and the overall visual style. No preamble, no opinions."
)

COMBINE_INSTRUCTION = (
    "You are writing a single text-to-image prompt. Combine the user's prompt with the reference "
    "details into ONE detailed prompt, LEADING with the user's prompt (it takes priority and keeps "
    "its subject). Weave the reference details in naturally. One flowing paragraph, no preamble, no lists."
)

# Everything that belongs to a person — named exhaustively so the filter pass leaves no residue
# (a short list lets hair, jewellery, tattoos, accessories, held objects, etc. slip through).
PERSON_ELEMENTS = (
    "people and EVERYTHING belonging to or associated with them — faces, facial features, expressions, "
    "gaze, hair, skin, hands, arms, legs, bodies, and poses; and all clothing, footwear, headwear, "
    "jewellery, watches, glasses, hats, scarves, gloves, bags, accessories, tattoos, piercings, "
    "makeup, nail polish, and anything they hold, carry, wear, or touch"
)

# Dropdown of what to REMOVE from the description (pass 2). Value = the removal clause (None = keep
# the full description). 'custom' (appended in the node) uses the custom_image_instruction box.
REMOVE_TARGETS = {
    "nothing (keep full description)": None,
    "people (and all they wear/hold)": "REMOVE " + PERSON_ELEMENTS + ". Keep the background, setting, "
        "and any non-person objects.",
    "the main subject": "REMOVE the main foreground subject and anything it holds, wears, or is composed "
        "of. Keep the background and setting.",
    "people + main subject (keep the scene)": "REMOVE " + PERSON_ELEMENTS + "; ALSO remove the main "
        "foreground subject and any objects it holds. Keep ONLY the background, setting, environment, "
        "architecture, landscape, lighting, weather, and atmosphere.",
    "the background (keep the subject)": "REMOVE the background, setting, and environment. Keep the main "
        "subject and its appearance.",
    "everything except clothing": "Keep ONLY the clothing and outfit (garments, footwear, colors, fabrics, "
        "patterns, accessories, jewellery). REMOVE the person's face, hair, skin, body, and pose, the "
        "background, and everything else.",
    "everything except style": "Keep ONLY the visual style (lighting, color palette, medium, mood, grain, "
        "rendering). REMOVE the specific subject and literal content.",
    "everything except colors": "Keep ONLY the color palette and overall tonality. REMOVE the subject, "
        "setting, and style.",
    "everything except composition": "Keep ONLY the composition (framing, camera angle, shot type, layout). "
        "REMOVE the specific subject identity, colors, and style.",
}


def _filter_instruction(clause):
    """Wrap a REMOVE_TARGETS clause (or a custom one) into a pass-2 text-editing instruction."""
    return ("You are editing an image description into a reusable fragment. " + clause.strip() +
            " Be thorough and literal: leave NOTHING that refers to the removed elements, not even "
            "indirectly (no 'where a person stood', no leftover shadows or reflections of them). "
            "Rewrite ONLY what remains as one clean, self-contained description. Do not mention what "
            "was removed and do not invent new details. No preamble.")


def _generate_text(clip, prompt, instruction, max_new_tokens, temperature, top_p, seed, image=None):
    """Run the Krea 2 encoder's own LLM to expand `prompt` under `instruction`. Returns text.

    Uses the CLIP's generate()/decode() (Qwen3-VL-4B has a real tied LM head, so this is genuine
    generation). thinking stays off -> the model answers directly instead of emitting <think> tags.
    When `image` is given, the vision tower reads it (works on bf16 and fp8_scaled encoders).
    """
    instr = (instruction or "").strip() or REASON_INSTRUCTION
    if image is not None:
        tmpl = ("<|im_start|>system\n" + _esc(instr) + "<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n")
        tokens = clip.tokenize(prompt, image=image, llama_template=tmpl)
    else:
        tmpl = ("<|im_start|>system\n" + _esc(instr) + "<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
        tokens = clip.tokenize(prompt, llama_template=tmpl)
    do_sample = float(temperature) > 0.0
    ids = clip.generate(
        tokens, do_sample=do_sample, max_length=int(max_new_tokens),
        temperature=float(temperature) if do_sample else 1.0,
        top_k=64, top_p=float(top_p), min_p=0.0, repetition_penalty=1.05,
        presence_penalty=0.0, seed=int(seed),
    )
    return " ".join(clip.decode(ids).split()).strip()


class KreaReason:
    """Reason-then-encode: the encoder's own LLM expands your prompt (or reads a reference image),
    then encodes the result.

    Krea 2's text encoder is a full Qwen3-VL-4B with a working LM head, so it can rewrite a short
    prompt into a rich one BEFORE conditioning — a built-in prompt-enhancer running on the exact
    model that conditions, no second LLM needed.

    mode:
      expand  — the generated, expanded prompt is what gets encoded (the impactful path).
      think   — the original prompt is encoded with the generated text placed in the <think>
                block as appended context (Qwen is causal, so this adds conditioning tokens after
                the prompt rather than rewriting it — subtler, more experimental).

    Optionally connect a reference `image` — a 3-pass pipeline runs: (1) the Qwen3-VL vision tower
    DESCRIBES the image in full; (2) a text pass FILTERS that description, removing the elements
    `image_remove` selects (people, subject, background, or "everything except <aspect>"); (3) a
    text pass COMBINES the filtered description with the user's prompt. Vision runs on both the bf16
    and (on ComfyUI) the fp8_scaled encoder; if a build can't run the vision tower the node raises a
    clear error suggesting bf16. `image_megapixels` caps the reference resolution. Costs 3 generations,
    so it's slower — a quality path, not a live dial.

    `cond_boost` multiplies the output conditioning (a CFG-free guidance boost; ~1.5-4x often sharpens
    Krea 2, saturates past ~6-8x). `generated_text` is exposed so you can preview/reuse what it wrote.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "mode": (["expand", "think"], {"default": "expand",
                         "tooltip": "expand: encode the rewritten prompt (impactful). think: keep prompt, append the reasoning as context (subtle/experimental). Ignored when an image is connected"}),
                "max_new_tokens": ("INT", {"default": 220, "min": 16, "max": 2048, "tooltip": "Length cap for each generation pass"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "0 = deterministic (greedy). Higher = more varied phrasing"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Sampling seed (only matters when temperature > 0)"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional reference image — described, filtered, then combined with your prompt (3 passes). Vision works on bf16 and fp8_scaled encoders; if a build can't run vision you'll get a clear error"}),
                "image_remove": (list(REMOVE_TARGETS.keys()) + ["custom"], {"default": "people + main subject (keep the scene)",
                                 "tooltip": "What to REMOVE from the image description before combining with your prompt. e.g. remove people + subject to keep only the scene/background. 'custom' uses the box below. Ignored if no image"}),
                "custom_image_instruction": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "Used only when image_remove = 'custom'. Say what to remove (or keep), e.g. 'remove the sky and any text'"}),
                "image_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1,
                                               "tooltip": "Downscale cap for the reference image (megapixels). Lower = fewer vision tokens / faster"}),
                "instruction": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "System instruction for text-only expansion (blank = built-in). Ignored when an image is connected (image_remove drives it)"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cond_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1,
                                         "tooltip": "Multiply the output conditioning (CFG-free guidance boost). 1.0 = off; ~1.5-4x often sharpens Krea 2; saturates past ~6-8x. Sweep by eye"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "generated_text")
    FUNCTION = "reason"
    CATEGORY = CATEGORY

    def reason(self, clip, prompt, mode, max_new_tokens, temperature, seed, image=None,
               image_remove="people + main subject (keep the scene)", custom_image_instruction="",
               image_megapixels=1.0, instruction="", top_p=0.95, cond_boost=1.0):
        has_img = image is not None
        up = (prompt or "").strip()
        if not up and not has_img:
            raise ValueError("[KreaReason] needs a prompt and/or a reference image")

        if has_img:
            img = _cap_image_tensor(image, image_megapixels)
            # Pass 1 (vision): full, faithful description of the image
            try:
                desc = _generate_text(clip, "Describe this image in complete detail.",
                                      DESCRIBE_FULL_INSTRUCTION, max_new_tokens, temperature, top_p, seed, image=img)
            except Exception as ex:
                raise RuntimeError(
                    "[KreaReason] vision path failed — this encoder build couldn't run the "
                    "Qwen3-VL vision tower. Try the bf16 encoder (qwen3vl_4b_bf16). "
                    f"Underlying error: {ex}")
            if not desc:
                raise RuntimeError("[KreaReason] the vision model returned no description")

            # Pass 2 (text): filter the description by removing the selected elements
            if image_remove == "custom":
                clause = (custom_image_instruction or "").strip()
            else:
                clause = REMOVE_TARGETS.get(image_remove)
            if clause:
                filtered = _generate_text(clip, desc, _filter_instruction(clause),
                                          max_new_tokens, temperature, top_p, seed)
            else:
                filtered = desc  # "nothing" / empty custom -> keep the full description

            # Pass 3 (text): combine the filtered reference with the user's prompt
            if up:
                combine_in = f"User's prompt: {up}\n\nReference details: {filtered}"
                text = _generate_text(clip, combine_in, COMBINE_INSTRUCTION,
                                      max_new_tokens, temperature, top_p, seed)
            else:
                text = filtered  # no user prompt -> just use the filtered reference

            print(f"[KreaReason] (image, remove='{image_remove}'): "
                  f"describe {len(desc.split())}w -> filter {len(filtered.split())}w -> final {len(text.split())}w\n{text}")
            cond = _encode(clip, text)
        else:
            text = _generate_text(clip, prompt, instruction, max_new_tokens, temperature, top_p, seed)
            if not text:
                print("[KreaReason] generation returned empty — falling back to the raw prompt")
                text = prompt
            print(f"[KreaReason] ({mode}) -> {len(text.split())} words:\n{text}")
            if mode == "think":
                cond = _encode(clip, prompt, reasoning=text)
            else:  # expand
                cond = _encode(clip, text)

        return (_scale_conditioning(cond, cond_boost), text)


NODE_CLASS_MAPPINGS = {
    "KreaReason": KreaReason,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaReason": "Krea Reason (expand prompt + encode)",
}

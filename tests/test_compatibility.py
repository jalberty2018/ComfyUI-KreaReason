"""CPU-only orchestration tests; no model weights or PyTorch required."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("kreareason_test_nodes", Path(__file__).parents[1] / "nodes.py")
nodes = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}):
    spec.loader.exec_module(nodes)


class Model:
    fixed_kv = True
    graph_dynamic_vbar_blocks = True
    prefetch_dynamic_vbars = True


class Clip:
    def __init__(self, texts=None, error=None):
        self.model = Model()
        self.model.fixed_kv = True
        self.cond_stage_model = SimpleNamespace(qwen3vl_4b=SimpleNamespace(
            transformer=SimpleNamespace(model=self.model)))
        self.texts = iter(texts or ["A sunlit garden."])
        self.error = error
        self.calls = []
        self.encodes = []

    def tokenize(self, text, **kwargs):
        return {"text": text, **kwargs}

    def generate(self, tokens, **kwargs):
        self.calls.append((tokens, kwargs, (self.model.fixed_kv,
                           self.model.graph_dynamic_vbar_blocks,
                           self.model.prefetch_dynamic_vbars)))
        if self.error:
            raise self.error
        return [1, 2]

    def decode(self, ids):
        return next(self.texts)

    def encode_from_tokens_scheduled(self, tokens):
        self.encodes.append(tokens)
        assert self.model.fixed_kv and self.model.graph_dynamic_vbar_blocks
        return [["conditioning", {}]]


class CompatibilityTests(unittest.TestCase):
    def generate(self, clip, **kwargs):
        return nodes._generate_text(clip, "garden", "", 220, 0.7, 0.95, 123, **kwargs)

    def test_compatible_restores_instance_and_class_attributes(self):
        clip = Clip()
        before = vars(clip.model).copy()
        self.assertEqual(self.generate(clip), "A sunlit garden.")
        self.assertEqual(clip.calls[0][2], (False, False, False))
        self.assertEqual(vars(clip.model), before)
        self.assertFalse(clip.calls[0][0]["thinking"])

    def test_native_preserves_optimizations(self):
        clip = Clip()
        self.generate(clip, generation_backend="native")
        self.assertEqual(clip.calls[0][2], (True, True, True))

    def test_cuda_failure_restores_flags_and_does_not_retry(self):
        error = RuntimeError("CUDA error: device-side assert triggered")
        clip = Clip(error=error)
        before = vars(clip.model).copy()
        with self.assertRaisesRegex(RuntimeError, "Restart ComfyUI") as caught:
            self.generate(clip)
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(len(clip.calls), 1)
        self.assertEqual(vars(clip.model), before)

    def test_unknown_future_api_restores_partial_changes(self):
        clip = Clip()
        clip.model = SimpleNamespace(fixed_kv=True)
        clip.cond_stage_model.qwen3vl_4b.transformer.model = clip.model
        with self.assertRaisesRegex(RuntimeError, "Unsupported ComfyUI"):
            self.generate(clip)
        self.assertEqual(vars(clip.model), {"fixed_kv": True})
        self.assertFalse(clip.calls)

    def test_wrong_encoder(self):
        clip = Clip()
        clip.cond_stage_model = None
        with self.assertRaisesRegex(ValueError, "CLIPLoader"):
            self.generate(clip)

    def test_empty_text_stops_before_encoding(self):
        clip = Clip(texts=["  "])
        with self.assertRaisesRegex(RuntimeError, "empty text"):
            nodes.KreaReason().reason(clip, "garden", "expand", 220, 0.7, 0)
        self.assertFalse(clip.encodes)

    def test_legacy_workflow_uses_compatible_default(self):
        clip = Clip()
        cond, text = nodes.KreaReason().reason(clip, "garden", "expand", 220, 0.7, 0)
        self.assertEqual(text, "A sunlit garden.")
        self.assertEqual(clip.encodes[0]["text"], text)
        self.assertEqual(clip.calls[0][2], (False, False, False))

    def test_three_image_passes_use_selected_backend(self):
        clip = Clip(texts=["A person in a garden.", "A garden.", "A cat in a garden."])
        with patch.object(nodes, "_cap_image_tensor", return_value="image"):
            nodes.KreaReason().reason(clip, "cat", "expand", 220, 0.7, 0,
                                     image="image", generation_backend="native")
        self.assertEqual(len(clip.calls), 3)
        self.assertTrue(all(call[2] == (True, True, True) for call in clip.calls))
        self.assertEqual(clip.calls[0][0]["image"], "image")
        self.assertEqual(clip.calls[1][0]["text"], "A person in a garden.")
        self.assertIn("A garden.", clip.calls[2][0]["text"])

    def test_greedy_and_think_preserve_original_prompt(self):
        clip = Clip()
        nodes.KreaReason().reason(clip, "garden", "think", 220, 0, 0)
        self.assertFalse(clip.calls[0][1]["do_sample"])
        self.assertEqual(clip.encodes[0]["text"], "garden")
        self.assertIn("A sunlit garden.", clip.encodes[0]["llama_template"])


if __name__ == "__main__":
    unittest.main()

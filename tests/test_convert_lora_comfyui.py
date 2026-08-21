"""Tests for scripts/convert_lora_comfyui.py.

Synthetic adapters match the LoRANetwork key layout this repo ships
(``lora_unet-transformer_blocks-N-attn-to_*``, ``lora_te-model-layers-N-…``).
A live test converts a real Hub transformer slider when huggingface_hub can
reach ntc-ai/minimax-music3-concept-sliders.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import convert_lora_comfyui as conv  # noqa: E402

HUB_REPO = "ntc-ai/minimax-music3-concept-sliders"
HUB_TF = "weights/energy-slider-v2/energy_unit_last.safetensors"
HUB_LM = "weights/gender-lm-v4/gender-lm-v4_last.safetensors"


def _write(path: Path, tensors: dict) -> Path:
    save_file(tensors, str(path))
    return path


def _tf_module(layer: int, tail: str, rank=4, dim=16, alpha=8.0):
    name = f"lora_unet-transformer_blocks-{layer}-attn-{tail}"
    return {
        f"{name}.lora_down.weight": torch.randn(rank, dim),
        f"{name}.lora_up.weight": torch.randn(dim, rank),
        f"{name}.alpha": torch.tensor(alpha),
    }


def _lm_module(layer: int, proj: str, rank=4, in_dim=32, out_dim=32, alpha=8.0):
    name = f"lora_te-model-layers-{layer}-self_attn-{proj}"
    return {
        f"{name}.lora_down.weight": torch.randn(rank, in_dim),
        f"{name}.lora_up.weight": torch.randn(out_dim, rank),
        f"{name}.alpha": torch.tensor(alpha),
    }


class MappingTests(unittest.TestCase):
    def test_transformer_attention_maps_to_comfy_dit_names(self):
        self.assertEqual(
            conv.map_transformer_module("lora_unet-transformer_blocks-3-attn-to_out-0"),
            "diffusion_transformer.transformer.layers.3.self_attn.to_out",
        )
        self.assertEqual(
            conv.map_transformer_module("lora_unet-transformer_blocks-3-attn-to_q"),
            ("qkv", "3", "q"),
        )
        self.assertEqual(
            conv.map_transformer_module("lora_unet-proj_in"),
            "diffusion_transformer.transformer.project_in",
        )
        self.assertEqual(
            conv.map_transformer_module("lora_unet-transformer_blocks-1-ff_in"),
            "diffusion_transformer.transformer.layers.1.ff.ff.0.proj",
        )
        self.assertIsNone(conv.map_transformer_module("lora_unet-not-a-real-module"))

    def test_lm_maps_to_text_encoders_qwen3_names(self):
        self.assertEqual(
            conv.map_lm_module("lora_te-model-layers-2-self_attn-q_proj"),
            "model.layers.2.self_attn.q_proj",
        )
        self.assertEqual(
            conv.map_lm_module("lora_te-model-layers-2-self_attn-o_proj"),
            "model.layers.2.self_attn.o_proj",
        )
        self.assertIsNone(conv.map_lm_module("lora_te-model-layers-2-mlp-gate_proj"))

    def test_detect_kind(self):
        self.assertEqual(
            conv.detect_kind(["lora_unet-transformer_blocks-0-attn-to_q.lora_down.weight"]),
            "transformer",
        )
        self.assertEqual(
            conv.detect_kind(["lora_te-model-layers-0-self_attn-q_proj.lora_down.weight"]),
            "language_model",
        )
        self.assertEqual(
            conv.detect_kind(["diffusion_model.diffusion_transformer.x.lora_A.weight"]),
            "already_comfyui",
        )

    def test_peft_keys_are_refused(self):
        with self.assertRaises(conv.ConvertError) as ctx:
            conv.detect_kind(
                ["base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight"]
            )
        self.assertIn("PEFT", str(ctx.exception))


class ConvertTests(unittest.TestCase):
    def test_transformer_fuses_qkv_and_keeps_to_out(self):
        tensors = {}
        for layer in (0, 1):
            for tail in ("to_q", "to_k", "to_v", "to_out-0"):
                tensors.update(_tf_module(layer, tail, rank=4, dim=16, alpha=8.0))
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(Path(tmp) / "slider.safetensors", tensors)
            kind, out, dropped = conv.convert_file(src)
        self.assertEqual(kind, "transformer")
        self.assertEqual(dropped, [])
        expected = {
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight",
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_B.weight",
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.alpha",
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_out.lora_A.weight",
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_out.lora_B.weight",
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_out.alpha",
            "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_qkv.lora_A.weight",
            "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_qkv.lora_B.weight",
            "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_qkv.alpha",
            "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_out.lora_A.weight",
            "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_out.lora_B.weight",
            "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_out.alpha",
        }
        self.assertEqual(set(out), expected)
        qkv_down = out[
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight"
        ]
        qkv_up = out[
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_B.weight"
        ]
        self.assertEqual(tuple(qkv_down.shape), (12, 16))  # 3 * rank
        self.assertEqual(tuple(qkv_up.shape), (48, 12))  # 3 * dim, 3 * rank
        self.assertEqual(
            float(out["diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.alpha"]),
            24.0,  # scale 2 * total_rank 12
        )

    def test_qkv_fusion_preserves_delta(self):
        rank, dim, alpha = 3, 8, 6.0
        q_down = torch.randn(rank, dim)
        k_down = torch.randn(rank, dim)
        v_down = torch.randn(rank, dim)
        q_up = torch.randn(dim, rank)
        k_up = torch.randn(dim, rank)
        v_up = torch.randn(dim, rank)
        tensors = {}
        for tail, down, up in (
            ("to_q", q_down, q_up),
            ("to_k", k_down, k_up),
            ("to_v", v_down, v_up),
        ):
            name = f"lora_unet-transformer_blocks-0-attn-{tail}"
            tensors[f"{name}.lora_down.weight"] = down
            tensors[f"{name}.lora_up.weight"] = up
            tensors[f"{name}.alpha"] = torch.tensor(alpha)
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(Path(tmp) / "qkv.safetensors", tensors)
            _kind, out, _dropped = conv.convert_file(src)
        fused_down, fused_up, fused_alpha = conv._block_diag_up(
            [
                (q_down, q_up, alpha),
                (k_down, k_up, alpha),
                (v_down, v_up, alpha),
            ]
        )
        scale = fused_alpha / fused_down.shape[0]
        fused_delta = (fused_up * scale) @ fused_down
        want = torch.cat(
            [
                q_up @ q_down * (alpha / rank),
                k_up @ k_down * (alpha / rank),
                v_up @ v_down * (alpha / rank),
            ],
            dim=0,
        )
        self.assertTrue(torch.allclose(fused_delta, want, atol=1e-5))
        # File write is bf16; the applied delta must still match at that precision.
        written_down = out[
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight"
        ].float()
        written_up = out[
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_B.weight"
        ].float()
        written_alpha = float(
            out["diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.alpha"]
        )
        written_delta = (written_up * (written_alpha / written_down.shape[0])) @ written_down
        self.assertTrue(torch.allclose(written_delta, want, atol=2e-2, rtol=1e-2))

    def test_full_targets_root_modules(self):
        tensors = _tf_module(0, "to_q")
        tensors.update(_tf_module(0, "to_k"))
        tensors.update(_tf_module(0, "to_v"))
        tensors.update(
            {
                "lora_unet-proj_in.lora_down.weight": torch.randn(4, 32),
                "lora_unet-proj_in.lora_up.weight": torch.randn(16, 4),
                "lora_unet-proj_in.alpha": torch.tensor(4.0),
                "lora_unet-preprocess_conv.lora_down.weight": torch.randn(4, 32),
                "lora_unet-preprocess_conv.lora_up.weight": torch.randn(32, 4),
                "lora_unet-preprocess_conv.alpha": torch.tensor(4.0),
                "lora_unet-transformer_blocks-0-ff_in.lora_down.weight": torch.randn(4, 16),
                "lora_unet-transformer_blocks-0-ff_in.lora_up.weight": torch.randn(64, 4),
                "lora_unet-transformer_blocks-0-ff_in.alpha": torch.tensor(4.0),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(Path(tmp) / "full.safetensors", tensors)
            _kind, out, dropped = conv.convert_file(src)
        self.assertEqual(dropped, [])
        self.assertIn(
            "diffusion_model.diffusion_transformer.transformer.project_in.lora_A.weight",
            out,
        )
        self.assertIn(
            "diffusion_model.diffusion_transformer.preprocess_conv.lora_A.weight",
            out,
        )
        self.assertIn(
            "diffusion_model.diffusion_transformer.transformer.layers.0.ff.ff.0.proj.lora_A.weight",
            out,
        )

    def test_lm_keeps_gqa_shapes(self):
        tensors = {}
        tensors.update(_lm_module(0, "q_proj", out_dim=32))
        tensors.update(_lm_module(0, "k_proj", out_dim=8))
        tensors.update(_lm_module(0, "v_proj", out_dim=8))
        tensors.update(_lm_module(0, "o_proj", out_dim=32))
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(Path(tmp) / "lm.safetensors", tensors)
            kind, out, dropped = conv.convert_file(src)
        self.assertEqual(kind, "language_model")
        self.assertEqual(dropped, [])
        self.assertEqual(
            tuple(out["text_encoders.model.layers.0.self_attn.k_proj.lora_B.weight"].shape),
            (8, 4),
        )
        self.assertEqual(
            tuple(out["text_encoders.model.layers.0.self_attn.q_proj.lora_B.weight"].shape),
            (32, 4),
        )
        self.assertIn("text_encoders.model.layers.0.self_attn.o_proj.lora_A.weight", out)

    def test_unmapped_key_errors(self):
        tensors = _tf_module(0, "to_q")
        tensors.update(_tf_module(0, "to_k"))
        tensors.update(_tf_module(0, "to_v"))
        tensors["lora_unet-mystery-linear.lora_down.weight"] = torch.randn(4, 8)
        tensors["lora_unet-mystery-linear.lora_up.weight"] = torch.randn(8, 4)
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(Path(tmp) / "bad.safetensors", tensors)
            with self.assertRaises(conv.ConvertError) as ctx:
                conv.convert_file(src)
        self.assertIn("unmappable", str(ctx.exception).lower())
        self.assertIn("mystery", str(ctx.exception))

    def test_incomplete_qkv_errors(self):
        tensors = _tf_module(0, "to_q")
        tensors.update(_tf_module(0, "to_k"))
        # no to_v
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(Path(tmp) / "partial.safetensors", tensors)
            with self.assertRaises(conv.ConvertError):
                conv.convert_file(src)

    def test_cli_writes_and_skips_existing(self):
        tensors = {}
        for tail in ("to_q", "to_k", "to_v", "to_out-0"):
            tensors.update(_tf_module(0, tail))
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(Path(tmp) / "one.safetensors", tensors)
            rc = conv.main([str(src)])
            self.assertEqual(rc, 0)
            out = Path(tmp) / "one_comfyui.safetensors"
            self.assertTrue(out.is_file())
            rc = conv.main([str(src)])
            self.assertEqual(rc, 0)  # skip existing
            with safe_open(str(out), framework="pt") as handle:
                keys = list(handle.keys())
            self.assertTrue(any(k.endswith("to_qkv.lora_A.weight") for k in keys))
            self.assertTrue(all(k.startswith("diffusion_model.") for k in keys))


def _hub_available(filename: str) -> str | None:
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(HUB_REPO, filename)
    except Exception:
        return None


class HubSliderTests(unittest.TestCase):
    def test_real_transformer_slider(self):
        local = _hub_available(HUB_TF)
        if local is None:
            self.skipTest(f"could not download {HUB_TF}")
        kind, out, dropped = conv.convert_file(local)
        self.assertEqual(kind, "transformer")
        self.assertEqual(dropped, [])
        qkv = [
            k
            for k in out
            if k.endswith(".self_attn.to_qkv.lora_A.weight")
        ]
        outs = [
            k
            for k in out
            if k.endswith(".self_attn.to_out.lora_A.weight")
        ]
        self.assertEqual(len(qkv), 36)
        self.assertEqual(len(outs), 36)
        self.assertTrue(all(k.startswith("diffusion_model.diffusion_transformer.") for k in out))
        sample = out[qkv[0]]
        self.assertEqual(tuple(sample.shape), (24, 2048))  # 3 * rank 8
        self.assertEqual(sample.dtype, torch.bfloat16)
        # no leftover attn.to_q/k/v names — they must be fused
        self.assertFalse(any(".to_q.lora_" in k or ".to_k.lora_" in k for k in out))

    def test_real_lm_slider(self):
        local = _hub_available(HUB_LM)
        if local is None:
            self.skipTest(f"could not download {HUB_LM}")
        kind, out, dropped = conv.convert_file(local)
        self.assertEqual(kind, "language_model")
        self.assertEqual(dropped, [])
        q = [k for k in out if k.endswith(".self_attn.q_proj.lora_A.weight")]
        k = [k for k in out if k.endswith(".self_attn.k_proj.lora_A.weight")]
        o = [k for k in out if k.endswith(".self_attn.o_proj.lora_A.weight")]
        self.assertEqual(len(q), 36)
        self.assertEqual(len(k), 36)
        self.assertEqual(len(o), 36)
        self.assertTrue(all(name.startswith("text_encoders.model.layers.") for name in out))
        # GQA: k_proj up is 1024 x 8 on the real Qwen3-8B
        k_up = out["text_encoders.model.layers.0.self_attn.k_proj.lora_B.weight"]
        q_up = out["text_encoders.model.layers.0.self_attn.q_proj.lora_B.weight"]
        self.assertEqual(tuple(k_up.shape), (1024, 8))
        self.assertEqual(tuple(q_up.shape), (4096, 8))


if __name__ == "__main__":
    unittest.main()

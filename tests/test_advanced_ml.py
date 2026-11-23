"""
Tests for LoRA fine-tuning pipeline and scratch Transformer model.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from carter.finetuning.lora import LoRAConfig, prepare_instruction_dataset, run_lora_training
from carter.scratch_transformer.model import CharTokenizer, TransformerConfig, ScratchTransformer


class TestLoRAFinetuning:
    def test_prepare_instruction_dataset(self, tmp_path: Path):
        data = [
            {"instruction": "List files", "response": '{"tool": "list_files"}'},
            {"instruction": "", "response": "Empty"},
            {"instruction": "Read note", "response": "Note content"},
        ]
        out_file = tmp_path / "instructions.jsonl"
        saved_path = prepare_instruction_dataset(data, out_file)

        assert saved_path.exists()
        lines = [json.loads(l) for l in saved_path.read_text(encoding="utf-8").strip().split("\n")]
        assert len(lines) == 2
        assert lines[0]["instruction"] == "List files"

    def test_run_lora_training_missing_base_model(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Base model not found"):
            run_lora_training(tmp_path / "missing.gguf", tmp_path / "data.jsonl")

    def test_run_lora_training_manifest_generation(self, tmp_path: Path):
        dummy_base = tmp_path / "model.bin"
        dummy_base.write_bytes(b"dummy")

        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"instruction": "hi", "response": "hello"}\n', encoding="utf-8")

        out_dir = tmp_path / "adapters"
        meta = run_lora_training(dummy_base, data_file, output_dir=out_dir)

        assert meta["status"] in ("success", "deferred_missing_dependencies")
        assert (out_dir / "adapter_config.json").exists() or (out_dir / "training_manifest.json").exists()


class TestScratchTransformer:
    def test_char_tokenizer_encode_decode(self):
        tok = CharTokenizer("abc 123")
        encoded = tok.encode("a b 1")
        decoded = tok.decode(encoded)
        assert decoded == "a b 1"

    def test_transformer_forward_pass_shape(self):
        cfg = TransformerConfig(vocab_size=30, block_size=16, n_embd=32, n_head=2, n_layer=1)
        model = ScratchTransformer(cfg, seed=42)

        tokens = [1, 2, 3, 4]
        logits = model.forward(tokens)
        assert logits.shape == (4, 30)

    def test_transformer_autoregressive_generation(self):
        tok = CharTokenizer()
        cfg = TransformerConfig(vocab_size=tok.vocab_size, block_size=16, n_embd=32, n_head=2, n_layer=1)
        model = ScratchTransformer(cfg, seed=99)

        prompt = tok.encode("Hi")
        gen = model.generate(prompt, max_new_tokens=5, temperature=1.0)
        assert len(gen) == len(prompt) + 5
        text = tok.decode(gen)
        assert text.startswith("Hi")

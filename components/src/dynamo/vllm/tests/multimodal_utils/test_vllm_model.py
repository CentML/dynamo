# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.model."""

import json
from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils import model as model_mod
from dynamo.vllm.multimodal_utils.model import (
    ModelFamily,
    construct_qwen_decode_mm_data,
    resolve_model_family,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _fake_vllm_model_with_visual(visual):
    model_runner = SimpleNamespace(model=SimpleNamespace(visual=visual))
    worker = SimpleNamespace(model_runner=model_runner)
    driver_worker = SimpleNamespace(worker=worker)
    model_executor = SimpleNamespace(driver_worker=driver_worker)
    inner_core = SimpleNamespace(model_executor=model_executor)
    engine_core = SimpleNamespace(engine_core=inner_core)
    llm_engine = SimpleNamespace(engine_core=engine_core)
    return SimpleNamespace(llm_engine=llm_engine)


class TestMultiModalUtils:
    def test_construct_qwen_decode_mm_data(self):
        max_rounds = int(torch.finfo(torch.float16).max) + 2
        expected_image_grid_thw_tensor = torch.tensor([16, 16])
        for i in range(max_rounds):
            # Should not raise any exception
            try:
                mm_data = construct_qwen_decode_mm_data(
                    image_grid_thw=[16, 16],
                    embeddings_shape=[2, 1024],
                    request_id=str(i),
                )
            except Exception as e:
                pytest.fail(
                    f"construct_qwen_decode_mm_data raised {type(e).__name__} on round {i}: {e}"
                )
            assert "image" in mm_data
            assert "image_grid_thw" in mm_data["image"]
            assert "image_embeds" in mm_data["image"]
            assert torch.allclose(
                mm_data["image"]["image_grid_thw"], expected_image_grid_thw_tensor
            )
            # Embedding values are randomly genearted as placehodler, we only check the shape
            assert mm_data["image"]["image_embeds"].shape == (2, 1024)


class TestLoadVisionModel:
    def test_encoder_worker_uses_regular_vllm_engine_args(self, monkeypatch):
        captured_kwargs = {}
        visual = object()

        def fake_llm(*, chat_template=None, **kwargs):
            captured_kwargs.update(kwargs)
            captured_kwargs["chat_template"] = chat_template
            return _fake_vllm_model_with_visual(visual)

        engine_args = SimpleNamespace(
            model="Qwen/Qwen2-VL-2B-Instruct",
            enforce_eager=True,
            tensor_parallel_size=2,
            dtype="bfloat16",
            gpu_memory_utilization=0.73,
            kv_cache_memory_bytes=123456,
            max_model_len=2048,
            enable_prefix_caching=True,
            chat_template="/tmp/qwen-template.jinja",
            mm_encoder_only=False,
            enable_log_requests=True,
            dynamo_internal_only=True,
        )
        monkeypatch.setattr(model_mod, "LLM", fake_llm)
        monkeypatch.setattr(model_mod, "update_environment_variables", lambda _: None)

        result = model_mod.load_vision_model(
            "Qwen/Qwen2-VL-2B-Instruct", engine_args=engine_args
        )

        assert result is visual
        assert captured_kwargs["model"] == "Qwen/Qwen2-VL-2B-Instruct"
        assert captured_kwargs["enforce_eager"] is True
        assert captured_kwargs["tensor_parallel_size"] == 2
        assert captured_kwargs["dtype"] == "bfloat16"
        assert captured_kwargs["gpu_memory_utilization"] == 0.73
        assert captured_kwargs["kv_cache_memory_bytes"] == 123456
        assert captured_kwargs["max_model_len"] == 2048
        assert captured_kwargs["enable_prefix_caching"] is True
        assert captured_kwargs["chat_template"] == "/tmp/qwen-template.jinja"
        assert captured_kwargs["mm_encoder_only"] is True
        assert "enable_log_requests" not in captured_kwargs
        assert "dynamo_internal_only" not in captured_kwargs

    def test_encoder_worker_preserves_legacy_defaults_without_engine_args(
        self, monkeypatch
    ):
        captured_kwargs = {}
        visual = object()

        def fake_llm(**kwargs):
            captured_kwargs.update(kwargs)
            return _fake_vllm_model_with_visual(visual)

        monkeypatch.setattr(model_mod, "LLM", fake_llm)
        monkeypatch.setattr(model_mod, "update_environment_variables", lambda _: None)

        result = model_mod.load_vision_model(
            "Qwen/Qwen2-VL-2B-Instruct", enforce_eager=True
        )

        assert result is visual
        assert captured_kwargs["enforce_eager"] is True
        assert captured_kwargs["gpu_memory_utilization"] == 0.2
        assert captured_kwargs["kv_cache_memory_bytes"] == 1024 * 1024 * 64
        assert captured_kwargs["max_model_len"] == 1
        assert captured_kwargs["mm_encoder_only"] is True
        assert captured_kwargs["enable_prefix_caching"] is False


class TestResolveModelFamily:
    """Cases where resolution is determined entirely by the input string
    (no filesystem state needed). Filesystem-dependent cases live in
    `TestResolveModelFamilyOnDisk`."""

    @pytest.mark.parametrize(
        "model_name, expected",
        [
            pytest.param(
                "Qwen/Qwen2-VL-2B-Instruct",
                ModelFamily.QWEN_VL,
                id="hf-id-qwen2-vl",
            ),
            pytest.param(
                "Qwen/Qwen3-VL-2B-Instruct",
                ModelFamily.QWEN_VL,
                id="hf-id-qwen3-vl",
            ),
            pytest.param(
                "Qwen/Qwen3.5-9B",
                ModelFamily.QWEN_VL,
                id="hf-id-qwen3.5-unified",
            ),
            pytest.param(
                "llava-hf/llava-1.5-7b-hf",
                ModelFamily.LLAVA,
                id="hf-id-llava",
            ),
            pytest.param(
                "/root/.cache/huggingface/hub/"
                "models--Qwen--Qwen2-VL-2B-Instruct/snapshots/abc123",
                ModelFamily.QWEN_VL,
                id="hf-cache-snapshot",
            ),
            pytest.param(
                "/local_store/Qwen--Qwen3-VL-2B-Instruct/v2",
                ModelFamily.QWEN_VL,
                id="local_store-parent-with-version",
            ),
            pytest.param(
                "/local_store/qwen2.5-vl-7b-instruct/v3",
                ModelFamily.QWEN_VL,
                id="local_store-org-less",
            ),
            pytest.param("RandomOrg/RandomModel-7B", None, id="unsupported-hf-id"),
        ],
    )
    def test_resolve_string_inputs(self, model_name, expected):
        assert resolve_model_family(model_name) == expected


class TestResolveModelFamilyOnDisk:
    """Cases that genuinely require filesystem state (a real `config.json` to
    exercise the metadata stage). Cases where directory existence is irrelevant
    to the result are covered string-only in `TestResolveModelFamily`."""

    @pytest.mark.parametrize(
        "subdir, architectures, expected",
        [
            pytest.param(
                "Qwen--Qwen2-VL-2B-Instruct/v2",
                ["Qwen2VLForConditionalGeneration"],
                ModelFamily.QWEN_VL,
                id="metadata-qwen2-vl",
            ),
            pytest.param(
                "Qwen--Qwen3-VL-2B-Instruct/v2",
                ["Qwen3VLForConditionalGeneration"],
                ModelFamily.QWEN_VL,
                id="metadata-qwen3-vl",
            ),
            pytest.param(
                "llava-hf--llava-1.5-7b-hf/v1",
                ["LlavaForConditionalGeneration"],
                ModelFamily.LLAVA,
                id="metadata-llava",
            ),
            pytest.param(
                "Qwen--Qwen3.5-9B/v1",
                ["Qwen3_5ForConditionalGeneration"],
                ModelFamily.QWEN_VL,
                id="metadata-qwen3.5-unified",
            ),
        ],
    )
    def test_metadata_stage_resolves_family(
        self, tmp_path, subdir, architectures, expected
    ):
        model_dir = tmp_path / subdir
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": architectures})
        )
        assert resolve_model_family(str(model_dir)) == expected

    def test_unrecognized_arch_falls_through_to_name_stage(self, tmp_path):
        """`config.json` exists but its arch isn't in the registry — the
        resolver must fall through to the name stage rather than return
        None on metadata miss."""
        model_dir = tmp_path / "Qwen--Qwen2-VL-2B-Instruct" / "v2"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": ["SomeFutureQwenVariantClass"]})
        )
        assert resolve_model_family(str(model_dir)) == ModelFamily.QWEN_VL

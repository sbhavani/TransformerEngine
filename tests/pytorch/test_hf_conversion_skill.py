# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Exercise skill conversion mechanics on CPU and real TE replacements on CUDA."""

import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn


SCRIPT = (
    Path(__file__).resolve().parents[2] / "skills/hf-transformer-engine/scripts/convert_model.py"
)
SPEC = importlib.util.spec_from_file_location("hf_te_converter", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)

AMPLIFY_SPEC = importlib.util.spec_from_file_location(
    "hf_te_amplify", SCRIPT.with_name("convert_amplify.py")
)
amplify_converter = importlib.util.module_from_spec(AMPLIFY_SPEC)
AMPLIFY_SPEC.loader.exec_module(amplify_converter)

FUSED_SPEC = importlib.util.spec_from_file_location("hf_te_fused", SCRIPT.with_name("fused.py"))
fused_converter = importlib.util.module_from_spec(FUSED_SPEC)
FUSED_SPEC.loader.exec_module(fused_converter)


def test_inventory_preserves_model():
    """Discovery must not alter values, identity, mode, or random state."""
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.LayerNorm(32)).eval()
    before = copy.deepcopy(model.state_dict())
    parameters = list(model.parameters())
    random_state = torch.random.get_rng_state()
    entries = converter.inventory(model)
    assert [entry["path"] for entry in entries if entry["eligible"]] == ["0", "2"]
    assert not model.training
    assert all(old is new for old, new in zip(parameters, model.parameters()))
    assert torch.equal(random_state, torch.random.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("sharing", ["parameter", "storage", "module"])
def test_shared_weights_need_adapter(sharing):
    """Tied objects, distinct Parameters sharing storage, and module aliases remain intact."""
    model = nn.ModuleDict({"first": nn.Linear(16, 16), "second": nn.Linear(16, 16)})
    if sharing == "module":
        model["second"] = model["first"]
    elif sharing == "parameter":
        model["second"].weight = model["first"].weight
    else:
        model["second"].weight = nn.Parameter(model["first"].weight.detach())
    entries = converter.inventory(model)
    assert not any(entry["eligible"] for entry in entries)
    with pytest.raises(ValueError, match="shared"):
        converter.replace_modules(model, ["first"])


@pytest.mark.parametrize(
    "case",
    [
        "hook",
        "parameter_hook",
        "meta",
        "buffer",
        "no_affine",
        "no_bias",
        "multi_axis",
        "custom",
        "dispatch",
        "forward",
    ],
)
def test_nonstandard_contracts_need_adapter(case):
    """Reject changes that would lose behavior or assume unsupported tensor semantics."""
    model = nn.Sequential(nn.Linear(16, 16))
    if case == "hook":
        model[0].register_forward_hook(lambda *args: None)
    elif case == "parameter_hook":
        model[0].weight.register_hook(lambda gradient: gradient)
    elif case == "meta":
        model.to("meta")
    elif case == "buffer":
        model[0].register_buffer("extra", torch.ones(1))
    elif case == "no_affine":
        model[0] = nn.LayerNorm(16, elementwise_affine=False)
    elif case == "no_bias":
        model[0] = nn.LayerNorm(16, bias=False)
    elif case == "multi_axis":
        model[0] = nn.LayerNorm((4, 4))
    elif case == "custom":

        class CustomLinear(nn.Linear):
            """A custom class cannot be assumed to implement the standard contract."""

        model[0] = CustomLinear(16, 16)
    elif case == "dispatch":
        model.hf_device_map = {"": "cpu"}
    else:
        model[0].forward = lambda value: value
    assert not converter.inventory(model)[0]["eligible"]
    original = model[0]
    with pytest.raises(ValueError):
        converter.replace_modules(model, ["0"])
    assert model[0] is original


@pytest.mark.parametrize("paths", [[], ["missing"], ["0", "0"], ["0", "missing"]])
def test_invalid_selection_does_not_mutate(paths):
    """Validate the entire selection before attempting any constructor."""
    model = nn.Sequential(nn.Linear(16, 16))
    original = model[0]
    with pytest.raises(ValueError):
        converter.replace_modules(model, paths)
    assert model[0] is original


def test_failed_preparation_is_atomic(monkeypatch):
    """Failure after one prepared module must preserve the tree and caller RNG."""
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
    original = list(model.children())
    random_state = torch.random.get_rng_state()

    def prepare(module):
        torch.rand(3)
        if module is original[1]:
            raise RuntimeError("constructor failure")
        return nn.Identity()

    monkeypatch.setattr(converter, "_make_replacement", prepare)
    with pytest.raises(RuntimeError, match="constructor failure"):
        converter.replace_modules(model, ["0", "1"])
    assert all(before is after for before, after in zip(original, model.children()))
    assert torch.equal(random_state, torch.random.get_rng_state())


def test_cpu_conversion_does_not_claim_te_execution():
    """Mechanically eligible CPU parameters still cannot execute the CUDA conversion."""
    model = nn.Sequential(nn.Linear(16, 16))
    assert converter.inventory(model)[0]["eligible"]
    with pytest.raises(ValueError, match="CUDA"):
        converter.replace_modules(model, ["0"])


def test_cli_local_factory_and_no_overwrite(tmp_path):
    """A factory in the working directory is discoverable; existing reports survive."""
    (tmp_path / "model_setup.py").write_text(
        "from torch import nn\ndef load_model():\n    return nn.Sequential(nn.Linear(16, 16))\n"
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--factory",
        "model_setup:load_model",
        "--output",
        "plan.json",
    ]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    report = (tmp_path / "plan.json").read_text()
    assert json.loads(report)[0]["eligible"]
    result = subprocess.run(command, cwd=tmp_path, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert (tmp_path / "plan.json").read_text() == report


def _tiny_hf_model(family):
    """Construct a local random HF model; no Hub artifacts are needed."""
    transformers = pytest.importorskip("transformers")
    if family in ("bert", "geneformer"):
        config = transformers.BertConfig(
            vocab_size=20275 if family == "geneformer" else 64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_act="relu" if family == "geneformer" else "gelu",
            max_position_embeddings=4096 if family == "geneformer" else 512,
        )
        model = transformers.BertForMaskedLM(config)
    elif family == "gpt2":
        config = transformers.GPT2Config(vocab_size=64, n_embd=32, n_layer=1, n_head=2)
        model = transformers.GPT2LMHeadModel(config)
    elif family == "llama":
        config = transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
        )
        model = transformers.LlamaForCausalLM(config)
    elif family == "esm":
        config = transformers.EsmConfig(
            vocab_size=33,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            position_embedding_type="rotary",
            token_dropout=True,
            pad_token_id=1,
            mask_token_id=32,
        )
        model = transformers.EsmForMaskedLM(config)
    elif family == "qwen3":
        if not hasattr(transformers, "Qwen3Config"):
            pytest.skip("This Transformers version does not provide Qwen3Config")
        config = transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            tie_word_embeddings=True,
        )
        model = transformers.Qwen3ForCausalLM(config)
    elif family == "mixtral":
        config = transformers.MixtralConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=4,
            num_experts_per_tok=2,
            tie_word_embeddings=True,
            output_router_logits=True,
        )
        model = transformers.MixtralForCausalLM(config)
    else:
        config = transformers.T5Config(
            vocab_size=64,
            d_model=32,
            d_ff=64,
            d_kv=16,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
        )
        model = transformers.T5ForConditionalGeneration(config)
    return model.eval()


@pytest.mark.parametrize(
    "family", ["bert", "gpt2", "llama", "t5", "esm", "geneformer", "qwen3", "mixtral"]
)
def test_tiny_hf_model_inventory(family):
    """Exercise live architectures without network access or pretrained downloads."""
    model = _tiny_hf_model(family)
    inputs = torch.tensor([[3, 4, 5, 6]])
    kwargs = {"input_ids": inputs, "labels": inputs}
    with torch.no_grad():
        assert torch.isfinite(model(**kwargs).loss)
    entries = converter.inventory(model)
    assert any(entry["eligible"] for entry in entries)
    # HF task heads tied to embeddings must stay outside generic conversion.
    assert any("shared parameter" in (entry["reason"] or "") for entry in entries)
    if family == "gpt2":
        assert any("Conv1D" in entry["type"] and not entry["eligible"] for entry in entries)
    if family == "llama":
        assert any("RMSNorm" in entry["type"] and not entry["eligible"] for entry in entries)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real TE conversion requires a CUDA GPU")
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_te_conversion_numerics_and_reload(bias, dtype):
    """Check actual TE outputs, input/parameter grads, optimizer step, and reload."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unsupported on this GPU")
    model = nn.Sequential(nn.Linear(128, 128, bias=bias), nn.GELU(), nn.LayerNorm(128))
    model.to(device="cuda", dtype=dtype).eval()
    model[2].bias.requires_grad_(False)
    reference = copy.deepcopy(model)
    parameters = list(model.parameters())
    manifest = converter.replace_modules(model, ["0", "2"])
    assert len(manifest["converted"]) == 2
    assert not model.training and not model[0].training and not model[2].training
    assert all(old is new for old, new in zip(parameters, model.parameters()))
    x = torch.randn(4, 128, device="cuda", dtype=dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    output, ref_output = model(x), reference(ref_x)
    tolerance = (
        {"rtol": 2e-4, "atol": 2e-5} if dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-2}
    )
    torch.testing.assert_close(output, ref_output, **tolerance)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    ref_output.backward(upstream)
    torch.testing.assert_close(x.grad, ref_x.grad, **tolerance)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, **tolerance)
    restored = nn.Sequential(nn.Linear(128, 128, bias=bias), nn.GELU(), nn.LayerNorm(128))
    restored.to(device="cuda", dtype=dtype).eval()
    restored[2].bias.requires_grad_(False)
    converter.replace_modules(restored, ["0", "2"])
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x.detach()), model(x.detach()), **tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HF/TE parity requires a CUDA GPU")
@pytest.mark.parametrize(
    "family", ["bert", "gpt2", "llama", "t5", "esm", "geneformer", "qwen3", "mixtral"]
)
def test_tiny_hf_te_parity(family):
    """Compare real HF task outputs, loss, gradients, ties, and decoder caches."""
    model = _tiny_hf_model(family).cuda()
    reference = copy.deepcopy(model)
    selected = [entry["path"] for entry in converter.inventory(model) if entry["eligible"]]
    converter.replace_modules(model, selected)
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    inputs = torch.tensor([[3, 4, 5, 6]], device="cuda")
    output = model(input_ids=inputs, labels=inputs)
    expected = reference(input_ids=inputs, labels=inputs)
    tolerance = {"rtol": 2e-4, "atol": 2e-5}
    torch.testing.assert_close(output.logits, expected.logits, **tolerance)
    torch.testing.assert_close(output.loss, expected.loss, **tolerance)
    if family == "mixtral":
        torch.testing.assert_close(output.aux_loss, expected.aux_loss, **tolerance)
        for actual, original in zip(output.router_logits, expected.router_logits):
            torch.testing.assert_close(actual, original, **tolerance)
    output.loss.backward()
    expected.loss.backward()
    # Native Parameter objects retain the original names for these leaf swaps.
    for name, parameter in model.named_parameters():
        expected_gradient = reference.get_parameter(name).grad
        if expected_gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected_gradient, **tolerance)
    if family in ("gpt2", "llama", "qwen3", "mixtral"):
        with torch.no_grad():
            prefill = model(input_ids=inputs, use_cache=True)
            ref_prefill = reference(input_ids=inputs, use_cache=True)
            next_token = inputs[:, :1]
            decode = model(
                input_ids=next_token, past_key_values=prefill.past_key_values, use_cache=True
            )
            ref_decode = reference(
                input_ids=next_token, past_key_values=ref_prefill.past_key_values, use_cache=True
            )
            torch.testing.assert_close(decode.logits, ref_decode.logits, **tolerance)


def test_amplify_packed_projection_contract():
    """The parent must call both projections and preserve packed gate order."""
    torch.manual_seed(7)
    w12, w3 = nn.Linear(16, 48, bias=False), nn.Linear(24, 16, bias=False)
    packed = amplify_converter.PackedSwiGLU(w12, w3)
    value = torch.randn(2, 5, 16, requires_grad=True)
    gate, up = torch.nn.functional.linear(value, w12.weight).chunk(2, dim=-1)
    reference = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, w3.weight)
    called = set()
    hooks = [
        layer.register_forward_hook(lambda *args, name=name: called.add(name))
        for name, layer in (("w12", w12), ("w3", w3))
    ]
    try:
        actual = packed(value)
    finally:
        for hook in hooks:
            hook.remove()
    assert called == {"w12", "w3"}
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(reference.sum(), value, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual.sum(), value)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert packed.w12.weight is w12.weight and packed.w3.weight is w3.weight


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Already-TE detection needs the TE extension"
)
def test_already_te_model_is_unchanged():
    """Report a native TE model as already converted rather than converting it again."""
    import transformer_engine.pytorch as te

    model = nn.Sequential(te.Linear(128, 128), te.LayerNorm(128))
    original = list(model.parameters())
    entries = converter.inventory(model)
    assert entries and all(not entry["eligible"] for entry in entries)
    assert all(entry["reason"] == "already a Transformer Engine module" for entry in entries)
    assert all(before is after for before, after in zip(original, model.parameters()))


BIO_CHECKPOINTS = {
    "esm2": ("facebook/esm2_t6_8M_UR50D", "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"),
    "amplify": ("chandar-lab/AMPLIFY_120M", "d918a9e8c64dc43f6ec4996022c3deba14d66470"),
}


def _load_bio_model(family, dtype, config=None):
    """Load the pinned original model; AMPLIFY remote code is an explicit opt-in."""
    from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

    checkpoint, revision = BIO_CHECKPOINTS[family]
    if family == "amplify" and os.getenv("NVTE_TEST_AMPLIFY_REMOTE_CODE") != "1":
        raise RuntimeError("Inspect pinned AMPLIFY code, then set NVTE_TEST_AMPLIFY_REMOTE_CODE=1")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    if config is None:
        if family == "esm2":
            model, loading = EsmForMaskedLM.from_pretrained(
                checkpoint,
                revision=revision,
                torch_dtype=dtype,
                output_loading_info=True,
                attn_implementation="eager",
            )
        else:
            model, loading = AutoModel.from_pretrained(
                checkpoint,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=dtype,
                output_loading_info=True,
                code_revision=revision,
            )
        assert not loading.get("missing_keys"), loading
        assert not loading.get("unexpected_keys"), loading
        model._conversion_loading_info = loading
    elif family == "esm2":
        model = EsmForMaskedLM(config)
    else:
        model = AutoModel.from_config(config, trust_remote_code=True, code_revision=revision)
    return model.to(device="cuda", dtype=dtype).eval(), tokenizer


def _bio_inputs(tokenizer, family, long_sequences):
    """Use actual protein tokens, fixed masked labels, and unequal padded lengths."""
    sequences = ["MKTAYIAKQRQISFVKSHFSRQ", "GAVLIPFYWSTCMNQDEKRH" * 2]
    if long_sequences:
        sequences = [sequence * 3 for sequence in sequences]
    batch = tokenizer(sequences, padding=True, pad_to_multiple_of=8, return_tensors="pt")
    labels = torch.full_like(batch["input_ids"], -100)
    for index in range(len(sequences)):
        labels[index, 3] = batch["input_ids"][index, 3]
        batch["input_ids"][index, 3] = tokenizer.mask_token_id
    keep = batch["attention_mask"].bool().cuda()
    inputs = {
        "input_ids": batch["input_ids"].cuda(),
        "output_hidden_states": True,
        "output_attentions": True,
    }
    if family == "amplify":
        # This checkpoint's source requires additive masks, unlike the BioNeMo fork.
        inputs["attention_mask"] = torch.where(keep, 0.0, float("-inf")).to(torch.bfloat16)
    else:
        inputs["attention_mask"] = keep.long()
    return inputs, labels.cuda(), keep


def _compare_bio_tensor(
    actual,
    expected,
    dtype,
    metrics,
    name,
    gradient=False,
    oracle=None,
    packed=None,
    max_rounding_steps=1,
    reduction_scale=None,
):
    """Compare FP32 directly; judge BF16 against an independent FP32 error baseline."""
    assert actual.shape == expected.shape, name
    assert actual.dtype == expected.dtype, name
    tensor_dtype = expected.dtype
    # A large vocabulary matrix otherwise creates several FP32 CUDA temporaries
    # while all reference models/gradients are live. Compute its statistics on CPU.
    on_cpu = actual.numel() >= 1_000_000

    def comparison_value(value):
        value = value.detach()
        return value.cpu().float() if on_cpu else value.float()

    actual, expected = comparison_value(actual), comparison_value(expected)
    assert torch.isfinite(actual).all(), name
    assert torch.isfinite(expected).all(), name
    difference = actual - expected
    scale = expected.square().mean().sqrt().item()
    rmse = difference.square().mean().sqrt().item()
    relative = rmse / max(scale, 1e-12)
    metrics[name] = {
        "relative_rmse": relative,
        "max_absolute_error": difference.abs().max().item(),
        "reference_rms": scale,
        "comparison_device": str(actual.device),
    }
    if dtype == torch.bfloat16:
        assert oracle is not None, f"{name}: BF16 requires a separately evaluated FP32 reference"
        assert oracle.shape == actual.shape, name
        oracle = comparison_value(oracle)
        assert torch.isfinite(oracle).all(), name
        native_error = expected - oracle
        converted_error = actual - oracle
        # Set the budget from native arithmetic and output rounding only. The
        # candidate's error never determines its own acceptance threshold.
        unit_roundoff = torch.finfo(dtype).eps / 2
        native_rmse = native_error.square().mean().sqrt().item()
        native_max = native_error.abs().max().item()
        baseline_rmse, baseline_max = native_rmse, native_max
        if packed is not None:
            assert packed.shape == actual.shape and packed.dtype == tensor_dtype, name
            packed = comparison_value(packed)
            assert torch.isfinite(packed).all(), name
            packed_error = packed - oracle
            packed_rmse = packed_error.square().mean().sqrt().item()
            packed_max = packed_error.abs().max().item()
            baseline_rmse = max(baseline_rmse, packed_rmse)
            baseline_max = max(baseline_max, packed_max)
            metrics[name].update(
                {
                    "pytorch_packed_to_fp32_rmse": packed_rmse,
                    "pytorch_packed_to_fp32_max_error": packed_max,
                }
            )
        oracle_rms = oracle.square().mean().sqrt().item()
        oracle_max = oracle.abs().max().item()
        max_scale = oracle_max
        if reduction_scale is not None:
            # A cancelling reduction's small result is not its roundoff scale.
            max_scale = max(max_scale, reduction_scale.max().item())
            metrics[name]["reference_reduction_max_sum_abs"] = reduction_scale.max().item()
        rmse_budget = 2 * baseline_rmse + unit_roundoff * oracle_rms
        max_budget = 2 * baseline_max + max_rounding_steps * unit_roundoff * max_scale
        converted_rmse = converted_error.square().mean().sqrt().item()
        converted_max = converted_error.abs().max().item()
        metrics[name].update(
            {
                "native_to_fp32_rmse": native_rmse,
                "converted_to_fp32_rmse": converted_rmse,
                "fp32_rms": oracle_rms,
                "rmse_budget": rmse_budget,
                "native_to_fp32_max_error": native_max,
                "converted_to_fp32_max_error": converted_max,
                "max_error_budget": max_budget,
                "max_rounding_steps": max_rounding_steps,
                "policy": "2x-native-error-envelope-plus-BF16-rounding",
                "rmse_rounding_steps": 1,
            }
        )
        assert converted_rmse <= rmse_budget, (name, metrics[name])
        assert converted_max <= max_budget, (name, metrics[name])
        return
    if packed is not None:
        assert packed.shape == actual.shape and packed.dtype == tensor_dtype, name
        packed = comparison_value(packed)
        assert torch.isfinite(packed).all(), name
        packing_error = packed - expected
        packing_rmse = packing_error.square().mean().sqrt().item()
        packing_max = packing_error.abs().max().item()
        unit_roundoff = torch.finfo(dtype).eps / 2
        rmse_budget = 2 * packing_rmse + unit_roundoff * scale
        max_budget = 2 * packing_max + unit_roundoff * expected.abs().max().item()
        metrics[name].update(
            {
                "converted_to_fp32_rmse": rmse,
                "converted_to_fp32_max_error": difference.abs().max().item(),
                "pytorch_packed_to_fp32_rmse": packing_rmse,
                "pytorch_packed_to_fp32_max_error": packing_max,
                "rmse_budget": rmse_budget,
                "max_error_budget": max_budget,
                "policy": "2x-PyTorch-packing-error-plus-one-FP32-rounding",
            }
        )
        assert rmse <= rmse_budget, (name, metrics[name])
        assert difference.abs().max().item() <= max_budget, (name, metrics[name])
        return
    metrics[name]["relative_rmse_bound"] = 5e-4
    assert relative <= 5e-4, (name, metrics[name])
    if not gradient:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=3e-4,
            atol=3e-5,
            msg=lambda detail: f"{name}: {detail}; metrics={metrics[name]}",
        )


def test_bf16_numerical_budget():
    """Allow rounding in either direction, and prefer a candidate closer to FP32."""
    oracle = torch.linspace(-3, 3, 257)
    native = (oracle + 0.002).bfloat16()
    candidate = (oracle - 0.002).bfloat16()
    metrics = {}
    _compare_bio_tensor(candidate, native, torch.bfloat16, metrics, "rounded", oracle=oracle)
    _compare_bio_tensor(oracle.bfloat16(), native, torch.bfloat16, metrics, "closer", oracle=oracle)
    assert metrics["rounded"]["converted_to_fp32_rmse"] <= metrics["rounded"]["rmse_budget"]


def test_fp32_packing_budget():
    """Accept measured packing roundoff and a candidate closer to native FP32."""
    native = torch.linspace(-3, 3, 257)
    packed = native + 1e-6
    for candidate in (packed, native):
        _compare_bio_tensor(candidate, native, torch.float32, {}, "packing", packed=packed)


def test_fused_gradient_outlier_budget():
    """Account for backward stores without hiding sparse gradient corruption in RMSE."""
    oracle = torch.ones(65536)
    native = oracle.bfloat16()
    candidate = native.clone()
    candidate[17] += 2 * torch.finfo(torch.bfloat16).eps
    _compare_bio_tensor(
        candidate, native, torch.bfloat16, {}, "gradient", True, oracle=oracle, max_rounding_steps=8
    )
    candidate[17] = 1.125
    metrics = {}
    with pytest.raises(AssertionError):
        _compare_bio_tensor(
            candidate,
            native,
            torch.bfloat16,
            metrics,
            "gradient",
            True,
            oracle=oracle,
            max_rounding_steps=8,
        )
    assert metrics["gradient"]["converted_to_fp32_rmse"] < metrics["gradient"]["rmse_budget"]
    assert (
        metrics["gradient"]["converted_to_fp32_max_error"] > metrics["gradient"]["max_error_budget"]
    )


def test_cancelling_gradient_roundoff_scale():
    """Use unreduced operands to scale cancellation error without weakening RMSE."""
    signal = torch.linspace(-0.01, 0.01, 1024)
    contributions = torch.stack((torch.full_like(signal, 0.5), signal - 0.5))
    oracle = contributions.sum(0)
    scale = contributions.abs().sum(0)
    native = (oracle + 0.0002 * torch.sin(torch.arange(1024))).bfloat16()
    candidate = native.clone()
    candidate[17] += 0.004
    with pytest.raises(AssertionError):
        _compare_bio_tensor(
            candidate,
            native,
            torch.bfloat16,
            {},
            "norm_gradient",
            True,
            oracle=oracle,
            max_rounding_steps=4,
        )
    metrics = {}
    _compare_bio_tensor(
        candidate,
        native,
        torch.bfloat16,
        metrics,
        "norm_gradient",
        True,
        oracle=oracle,
        max_rounding_steps=4,
        reduction_scale=scale,
    )
    candidate.zero_()
    with pytest.raises(AssertionError):
        _compare_bio_tensor(
            candidate,
            native,
            torch.bfloat16,
            {},
            "norm_gradient",
            True,
            oracle=oracle,
            max_rounding_steps=4,
            reduction_scale=scale,
        )


@pytest.mark.parametrize("defect", ["zero_gradient", "gate_order", "outlier", "zero_reference"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_numerical_budget_rejects_defects(defect, dtype):
    """An empirical rounding budget must still detect material conversion errors."""
    oracle = torch.linspace(-3, 3, 257)
    native = oracle.to(dtype)
    candidate = native.clone()
    if defect == "zero_gradient":
        candidate.zero_()
    elif defect == "gate_order":
        gate, up = native, native.flip(0)
        oracle = torch.nn.functional.silu(gate.float()) * up.float()
        native = torch.nn.functional.silu(gate) * up
        candidate = torch.nn.functional.silu(up) * gate
    elif defect == "outlier":
        candidate[128] = 1
    else:
        native = torch.zeros_like(native)
        oracle = torch.zeros_like(oracle)
        candidate = torch.full_like(candidate, 1e-6)
    with pytest.raises(AssertionError):
        _compare_bio_tensor(candidate, native, dtype, {}, defect, True, oracle, packed=native)


class NativePackedSwiGLU(nn.Module):
    """Independent PyTorch baseline for the rounding introduced by packing GEMMs."""

    def __init__(self, native):
        super().__init__()
        self.w12, self.w3 = native.w12, native.w3

    def forward(self, value):
        """Evaluate the native weights with one packed GEMM and explicit slices."""
        projected = torch.nn.functional.linear(value, self.w12.weight, self.w12.bias)
        width = self.w3.in_features
        activated = torch.nn.functional.silu(projected[..., :width]) * projected[..., width:]
        return torch.nn.functional.linear(activated, self.w3.weight, self.w3.bias)


def test_native_packed_swiglu_reference():
    """Validate the rounding baseline's math and all gradients against separate FP64 GEMMs."""
    native = nn.Module()
    native.w12, native.w3 = nn.Linear(16, 48).double(), nn.Linear(24, 16).double()
    packed = NativePackedSwiGLU(native)
    value = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    gate = torch.nn.functional.linear(value, native.w12.weight[:24], native.w12.bias[:24])
    up = torch.nn.functional.linear(value, native.w12.weight[24:], native.w12.bias[24:])
    reference = native.w3(torch.nn.functional.silu(gate) * up)
    actual = packed(value)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    variables = (value, *native.parameters())
    for actual_grad, reference_grad in zip(
        torch.autograd.grad(actual.square().sum(), variables, retain_graph=True),
        torch.autograd.grad(reference.square().sum(), variables),
    ):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-12)


@pytest.fixture(name="bio_math_mode")
def _bio_math_mode():
    """Compare IEEE FP32 math, restoring the caller's settings after each case."""
    if torch.cuda.is_available():
        assert (
            os.getenv("NVIDIA_TF32_OVERRIDE") == "0"
        ), "Set NVIDIA_TF32_OVERRIDE=0 before starting Python for IEEE FP32 conversion checks"
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield {
            "NVIDIA_TF32_OVERRIDE": os.getenv("NVIDIA_TF32_OVERRIDE"),
            "pytorch_matmul_allow_tf32": False,
            "pytorch_cudnn_allow_tf32": False,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


@pytest.fixture(name="bio_evidence")
def _bio_evidence(request, tmp_path):
    """Retain partial comparison metrics even when a checkpoint case fails."""
    data = {"case": request.node.nodeid, "completed": False}
    yield data
    output = Path(os.getenv("NVTE_TEST_HF_BIO_RESULTS", str(tmp_path)))
    output.mkdir(parents=True, exist_ok=True)
    filename = data.get("file_name", request.node.name.replace("/", "_"))
    (output / f"{filename}.json").write_text(json.dumps(data, indent=2) + "\n")


@pytest.mark.skipif(
    os.getenv("NVTE_TEST_HF_BIO_PRETRAINED") != "1",
    reason="Opt-in pretrained downloads: set NVTE_TEST_HF_BIO_PRETRAINED=1",
)
@pytest.mark.parametrize(
    "family,dtype",
    [
        ("esm2", torch.float32),
        ("esm2", torch.bfloat16),
        ("amplify", torch.float32),
        ("amplify", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("long_sequences", [False, True])
def test_pretrained_bio_conversion(
    family, dtype, long_sequences, tmp_path, bio_math_mode, bio_evidence
):
    """Test real checkpoints, executed TE coverage, gradients, strict reload, and HF export."""
    assert torch.cuda.is_available(), "Pretrained conversion validation must execute on CUDA"
    bio_evidence["file_name"] = f"{family}-{str(dtype).split('.')[-1]}-{long_sequences}"
    if dtype == torch.float32:
        assert (
            os.getenv("NVIDIA_TF32_OVERRIDE") == "0"
        ), "Set NVIDIA_TF32_OVERRIDE=0 before Python starts to align Torch and TE FP32 GEMMs"
    original, tokenizer = _load_bio_model(family, dtype)
    # Promote the already-rounded checkpoint rather than loading different FP32
    # weights: this isolates arithmetic differences from weight quantization.
    oracle_model = copy.deepcopy(original).float() if dtype == torch.bfloat16 else None
    packed_model = None
    if family == "amplify":
        packed_model = copy.deepcopy(original)
        for block in packed_model.transformer_encoder:
            block.ffn = NativePackedSwiGLU(block.ffn)
            block.ffn.eval()
    converted = copy.deepcopy(original)
    original_parameters = dict(converted.named_parameters())
    native_keys = list(original.state_dict())
    if family == "amplify":
        manifest = amplify_converter.convert_amplify(converted)
        paths = manifest["converted"]
        assert len(paths) == original.config.num_hidden_layers * 6
    else:
        paths = [
            entry["path"]
            for entry in converter.inventory(converted)
            if entry["eligible"] and ".contact_head." not in entry["path"]
        ]
        required = {
            f"esm.encoder.layer.{index}.{suffix}"
            for index in range(original.config.num_hidden_layers)
            for suffix in (
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
                "intermediate.dense",
                "output.dense",
            )
        }
        assert required <= set(paths), f"Missing core projections: {required - set(paths)}"
        manifest = converter.replace_modules(converted, paths)
    assert set(original_parameters) == set(dict(converted.named_parameters()))
    for name, parameter in converted.named_parameters():
        assert parameter is original_parameters[name]
        torch.testing.assert_close(parameter, original.get_parameter(name), rtol=0, atol=0)

    inputs, labels, keep = _bio_inputs(tokenizer, family, long_sequences)
    if family == "amplify":
        inputs["attention_mask"] = inputs["attention_mask"].to(dtype)
    oracle_inputs = {
        key: (
            value.float()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
        )
        for key, value in inputs.items()
    }
    oracle_output = oracle_model(**oracle_inputs) if oracle_model is not None else None
    packed_output = packed_model(**inputs) if packed_model is not None else None
    called, metrics = set(), {}
    bio_evidence.update(
        {
            "model": BIO_CHECKPOINTS[family],
            "dtype": str(dtype),
            "input_shape": list(inputs["input_ids"].shape),
            "metrics": metrics,
            "converted_paths": paths,
            "manifest": manifest,
        }
    )
    hooks = [
        converted.get_submodule(path).register_forward_hook(
            lambda *args, path=path: called.add(path)
        )
        for path in paths
    ]
    try:
        actual, expected = converted(**inputs), original(**inputs)
    finally:
        for hook in hooks:
            hook.remove()
    assert called == set(paths), f"TE replacements bypassed by forward: {set(paths) - called}"
    _compare_bio_tensor(
        actual.logits[keep],
        expected.logits[keep],
        dtype,
        metrics,
        "logits",
        oracle=oracle_output.logits[keep] if oracle_output is not None else None,
        packed=packed_output.logits[keep] if packed_output is not None else None,
    )
    for index, (value, reference) in enumerate(zip(actual.hidden_states, expected.hidden_states)):
        _compare_bio_tensor(
            value[keep],
            reference[keep],
            dtype,
            metrics,
            f"hidden.{index}",
            oracle=oracle_output.hidden_states[index][keep] if oracle_output is not None else None,
            packed=packed_output.hidden_states[index][keep] if packed_output is not None else None,
        )
    for index, (value, reference) in enumerate(zip(actual.attentions, expected.attentions)):
        _compare_bio_tensor(
            value,
            reference,
            dtype,
            metrics,
            f"attention.{index}",
            oracle=oracle_output.attentions[index] if oracle_output is not None else None,
            packed=packed_output.attentions[index] if packed_output is not None else None,
        )
    actual_loss = torch.nn.functional.cross_entropy(
        actual.logits.float().flatten(0, 1), labels.flatten()
    )
    expected_loss = torch.nn.functional.cross_entropy(
        expected.logits.float().flatten(0, 1), labels.flatten()
    )
    oracle_loss = (
        torch.nn.functional.cross_entropy(oracle_output.logits.flatten(0, 1), labels.flatten())
        if oracle_output is not None
        else None
    )
    packed_loss = (
        torch.nn.functional.cross_entropy(
            packed_output.logits.float().flatten(0, 1), labels.flatten()
        )
        if packed_output is not None
        else None
    )
    _compare_bio_tensor(
        actual_loss, expected_loss, dtype, metrics, "loss", oracle=oracle_loss, packed=packed_loss
    )
    with torch.no_grad():
        length = int(keep[0].sum())
        unpadded = {"input_ids": inputs["input_ids"][:1, :length], "output_hidden_states": True}
        if family == "esm2":
            unpadded["attention_mask"] = torch.ones_like(unpadded["input_ids"])
        native_single, te_single = original(**unpadded), converted(**unpadded)
        _compare_bio_tensor(
            te_single.logits,
            native_single.logits,
            dtype,
            metrics,
            "unpadded_logits",
            oracle=oracle_model(**unpadded).logits if oracle_model is not None else None,
            packed=packed_model(**unpadded).logits if packed_model is not None else None,
        )
        # Preserve the native checkpoint's padding response; it is not invariant.
        native_delta = native_single.logits.float() - expected.logits[:1, :length].float()
        metrics["native_padding_response"] = {
            "max_absolute_change": native_delta.abs().max().item(),
            "rms_change": native_delta.square().mean().sqrt().item(),
        }
        if family == "esm2":
            _compare_bio_tensor(
                converted.predict_contacts(inputs["input_ids"], keep.long()),
                original.predict_contacts(inputs["input_ids"], keep.long()),
                dtype,
                metrics,
                "contacts",
                oracle=(
                    oracle_model.predict_contacts(inputs["input_ids"], keep.long())
                    if oracle_model is not None
                    else None
                ),
            )
    actual_loss.backward()
    expected_loss.backward()
    if oracle_loss is not None:
        oracle_loss.backward()
    if packed_loss is not None:
        packed_loss.backward()
    for name, parameter in converted.named_parameters():
        reference = original.get_parameter(name)
        if reference.grad is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            _compare_bio_tensor(
                parameter.grad,
                reference.grad,
                dtype,
                metrics,
                f"gradient.{name}",
                True,
                oracle=oracle_model.get_parameter(name).grad if oracle_model is not None else None,
                packed=packed_model.get_parameter(name).grad if packed_model is not None else None,
            )
    torch.optim.SGD(converted.parameters(), lr=0.01).step()
    torch.optim.SGD(original.parameters(), lr=0.01).step()
    if oracle_model is not None:
        torch.optim.SGD(oracle_model.parameters(), lr=0.01).step()
    if packed_model is not None:
        torch.optim.SGD(packed_model.parameters(), lr=0.01).step()
    for name, parameter in converted.named_parameters():
        _compare_bio_tensor(
            parameter,
            original.get_parameter(name),
            dtype,
            metrics,
            f"updated_weight.{name}",
            oracle=oracle_model.get_parameter(name) if oracle_model is not None else None,
            packed=packed_model.get_parameter(name) if packed_model is not None else None,
        )
    with torch.no_grad():
        _compare_bio_tensor(
            converted(**inputs).logits[keep],
            original(**inputs).logits[keep],
            dtype,
            metrics,
            "post_step_logits",
            oracle=oracle_model(**oracle_inputs).logits[keep] if oracle_model is not None else None,
            packed=packed_model(**inputs).logits[keep] if packed_model is not None else None,
        )

    # Reconstruct the same TE architecture before strict loading.
    checkpoint = tmp_path / "converted.pt"
    if family == "amplify":
        amplify_converter.save_converted(converted, manifest, checkpoint)
        restored = amplify_converter.load_converted(
            lambda: _load_bio_model(family, dtype, copy.deepcopy(original.config))[0], checkpoint
        )
    else:
        restored, _ = _load_bio_model(family, dtype, copy.deepcopy(original.config))
        converter.replace_modules(restored, paths)
        torch.save(converted.state_dict(), checkpoint)
        restored.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True
        )
    with torch.no_grad():
        torch.testing.assert_close(
            restored(**inputs).logits, converted(**inputs).logits, rtol=0, atol=0
        )

    # Export only explicitly accounted native keys; never send TE metadata into HF loading.
    state = converted.state_dict()
    extra = set(state) - set(native_keys)
    assert all(key.endswith("_extra_state") for key in extra), extra
    native_state = {key: state[key] for key in native_keys}
    if family == "amplify":
        native_state = amplify_converter.export_native_state(converted, manifest)
    original.load_state_dict(native_state, strict=True)
    if oracle_model is not None:
        # Export uses the converted updated weights, so re-anchor the arithmetic
        # reference to those exact weights rather than the independent SGD run.
        oracle_model.load_state_dict(native_state, strict=True)
    if packed_model is not None:
        packed_model.load_state_dict(native_state, strict=True)
    for name, parameter in converted.named_parameters():
        torch.testing.assert_close(parameter, original.get_parameter(name), rtol=0, atol=0)
    export = tmp_path / "hf_export"
    original.save_pretrained(export)
    tokenizer.save_pretrained(export)
    export_options = {"attn_implementation": "eager"} if family == "esm2" else {}
    exported = (
        type(original)
        .from_pretrained(export, torch_dtype=dtype, **export_options)
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    for name, parameter in original.named_parameters():
        torch.testing.assert_close(exported.get_parameter(name), parameter, rtol=0, atol=0)
    with torch.no_grad():
        _compare_bio_tensor(
            converted(**inputs).logits[keep],
            exported(**inputs).logits[keep],
            dtype,
            metrics,
            "hf_export_logits",
            oracle=oracle_model(**oracle_inputs).logits[keep] if oracle_model is not None else None,
            packed=packed_model(**inputs).logits[keep] if packed_model is not None else None,
        )
    bio_evidence.update(
        {
            "model": BIO_CHECKPOINTS[family],
            "dtype": str(dtype),
            "input_shape": list(inputs["input_ids"].shape),
            "converted_paths": paths,
            "executed_paths": sorted(called),
            "manifest": manifest,
            "checkpoint_loading": getattr(original, "_conversion_loading_info", None),
            "math_mode": bio_math_mode,
            "numerical_reference": (
                "FP32 native model with BF16-rounded checkpoint weights"
                if oracle_model is not None
                else "native FP32 model"
            ),
            "rounding_baseline": (
                "Independent PyTorch packed gate/up, validated against separate FP64 GEMMs"
                if packed_model is not None
                else None
            ),
            "metrics": metrics,
            "completed": True,
        }
    )


def _fusion_parts(operation, normalization, bias):
    if normalization == "RMSNorm" and not hasattr(nn, "RMSNorm"):
        pytest.skip("This PyTorch version does not provide nn.RMSNorm")
    norm = (
        nn.LayerNorm(128, eps=1e-6) if normalization == "LayerNorm" else nn.RMSNorm(128, eps=1e-6)
    )
    if operation == "linear":
        return nn.ModuleDict({"norm": norm, "linear": nn.Linear(128, 192, bias=bias)})
    return nn.ModuleDict(
        {
            "norm": norm,
            "gate": nn.Linear(128, 256, bias=bias),
            "up": nn.Linear(128, 256, bias=bias),
            "down": nn.Linear(256, 128, bias=bias),
        }
    )


def _native_fusion_forward(parts, operation, value):
    normalized = parts["norm"](value)
    if operation == "linear":
        return parts["linear"](normalized)
    return parts["down"](
        torch.nn.functional.silu(parts["gate"](normalized)) * parts["up"](normalized)
    )


def _reference_norm(norm, value):
    """Independent native reference for FP32 affine math before output rounding."""
    accumulation = torch.float64 if value.dtype == torch.float64 else torch.float32
    data, weight = value.to(accumulation), norm.weight.to(accumulation)
    eps = norm.variance_epsilon if hasattr(norm, "variance_epsilon") else norm.eps
    if getattr(norm, "bias", None) is not None:
        result = torch.nn.functional.layer_norm(
            data, (data.shape[-1],), weight, norm.bias.to(accumulation), eps
        )
    else:
        result = data * torch.rsqrt(data.square().mean(-1, keepdim=True) + eps) * weight
    return result.to(value.dtype)


def _reference_swiglu(gate, up, down, value):
    weight = torch.cat((gate.weight, up.weight), dim=0)
    bias = torch.cat((gate.bias, up.bias), dim=0) if gate.bias is not None else None
    projected = torch.nn.functional.linear(value, weight, bias)
    first, second = projected.chunk(2, dim=-1)
    accumulation = torch.float64 if value.dtype == torch.float64 else torch.float32
    activation = (torch.nn.functional.silu(first.to(accumulation)) * second.to(accumulation)).to(
        value.dtype
    )
    return torch.nn.functional.linear(activation, down.weight, down.bias)


def _reference_fusion_forward(parts, operation, value):
    normalized = _reference_norm(parts["norm"], value)
    if operation == "linear":
        return parts["linear"](normalized)
    return _reference_swiglu(parts["gate"], parts["up"], parts["down"], normalized)


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_fusion_reference_math(normalization):
    """Validate the fusion-rounding baseline independently against unfused FP64 math."""
    parts = _fusion_parts("mlp", normalization, True).double()
    value = torch.randn(2, 3, 128, dtype=torch.float64, requires_grad=True)
    actual = _reference_fusion_forward(parts, "mlp", value)
    expected = _native_fusion_forward(parts, "mlp", value)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    variables = (value, *parts.parameters())
    for result, reference in zip(
        torch.autograd.grad(actual.square().sum(), variables, retain_graph=True),
        torch.autograd.grad(expected.square().sum(), variables),
    ):
        torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("cpu", "CUDA"),
        ("norm", "affine weight"),
        ("shared", "shared"),
        ("hook", "hooks"),
        ("mode", "consistent"),
        ("dimension", "dimensions"),
        ("bias", "bias"),
        ("default_eps", "dtype-dependent epsilon"),
    ],
)
def test_fusion_rejects_unsupported_contracts(case, reason):
    """Reject unsupported boundaries before importing TE or mutating source state."""
    parts = _fusion_parts("mlp", "LayerNorm", False)
    if case == "norm":
        parts["norm"] = nn.Identity()
    elif case == "shared":
        parts["up"].weight = parts["gate"].weight
    elif case == "hook":
        parts["norm"].register_forward_hook(lambda *args: None)
    elif case == "mode":
        parts["norm"].eval()
    elif case == "dimension":
        parts["up"] = nn.Linear(128, 512, bias=False)
    elif case == "bias":
        parts["down"] = nn.Linear(256, 128, bias=True)
    elif case == "default_eps":
        if not hasattr(nn, "RMSNorm"):
            pytest.skip("This PyTorch version does not provide nn.RMSNorm")
        parts["norm"] = nn.RMSNorm(128)
    parameters = list(parts.parameters())
    with pytest.raises(ValueError, match=reason):
        fused_converter.build_norm_mlp(parts["norm"], parts["gate"], parts["up"], parts["down"])
    assert all(before is after for before, after in zip(parameters, parts.parameters()))


def _qwen_fixture(family, dtype=torch.float32):
    transformers = pytest.importorskip("transformers")

    name = "Qwen2" if family == "qwen2" else "Qwen3"
    if not hasattr(transformers, f"{name}Config"):
        pytest.skip(f"This Transformers version does not provide {name}Config")
    config = getattr(transformers, f"{name}Config").from_dict(
        {
            "vocab_size": 64,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "tie_word_embeddings": True,
            "attention_dropout": 0.0,
            "pad_token_id": 0,
        }
    )
    config._attn_implementation = "eager"
    return transformers.AutoModelForCausalLM.from_config(
        config, torch_dtype=dtype, attn_implementation="eager"
    ).eval()


def test_qwen_fusion_preparation_is_atomic(monkeypatch):
    """Constructor failure must not remove a norm or replace any earlier block."""
    model = _qwen_fixture("qwen3")
    modules = dict(model.named_modules())
    calls = []

    def prepare(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("constructor failure")
        return nn.Identity(), []

    monkeypatch.setattr(fused_converter, "build_norm_mlp", prepare)
    with pytest.raises(RuntimeError, match="constructor failure"):
        fused_converter.convert_qwen_mlp(model)
    assert calls and all(model.get_submodule(path) is module for path, module in modules.items())


def _fusion_gradients(module, mapping):
    gradients = {}
    for row in mapping:
        gradient = module.get_parameter(row["target"]).grad
        values = (
            gradient.split([shape[0] for shape in row["shapes"]], dim=0)
            if gradient is not None and len(row["sources"]) > 1
            else [gradient] * len(row["sources"])
        )
        gradients.update(zip(row["sources"], values))
    return gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE fused modules require CUDA")
@pytest.mark.parametrize("operation", ["linear", "mlp"])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_builders(operation, normalization, bias, dtype, bio_math_mode, bio_evidence):
    """Exercise real LNL/LNMLP forward/backward, frozen state, export, and strict reload."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unsupported on this GPU")
    parts = _fusion_parts(operation, normalization, bias).to(device="cuda", dtype=dtype).eval()
    bio_evidence["math_mode"] = bio_math_mode
    bio_evidence["file_name"] = (
        f"fused-{operation}-{normalization}-{bias}-{str(dtype).split('.')[-1]}"
    )
    with torch.no_grad():
        parts["norm"].weight.normal_(mean=1.0, std=0.1)
    parts["norm"].weight.requires_grad_(False)
    oracle = copy.deepcopy(parts).float()
    variant = copy.deepcopy(parts)
    builder = (
        fused_converter.build_norm_linear
        if operation == "linear"
        else fused_converter.build_norm_mlp
    )
    names = ("norm", "linear") if operation == "linear" else ("norm", "gate", "up", "down")
    random_cpu, random_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    fused, mapping = builder(*(parts[name] for name in names))
    assert type(fused).__name__ == ("LayerNormLinear" if operation == "linear" else "LayerNormMLP")
    assert torch.equal(random_cpu, torch.get_rng_state())
    assert torch.equal(random_cuda, torch.cuda.get_rng_state())
    assert not fused.training and not fused.layer_norm_weight.requires_grad
    manifest = {"mapping": mapping, "native_state_keys": list(parts.state_dict())}
    for key, value in fused_converter.export_native_state(fused, manifest).items():
        torch.testing.assert_close(value, parts.state_dict()[key], rtol=0, atol=0)
    value = torch.randn(2, 8, 128, device="cuda", dtype=dtype, requires_grad=True)
    native_input = value.detach().clone().requires_grad_(True)
    oracle_input = value.detach().float().requires_grad_(True)
    variant_input = value.detach().clone().requires_grad_(True)
    actual = fused(value)
    expected = _native_fusion_forward(parts, operation, native_input)
    high = _native_fusion_forward(oracle, operation, oracle_input)
    alternate = _reference_fusion_forward(variant, operation, variant_input)
    metrics = {}
    bio_evidence.update(
        {
            "dtype": str(dtype),
            "metrics": metrics,
            "mapping": mapping,
            "pretrained": False,
            "input_shape": list(value.shape),
        }
    )
    _compare_bio_tensor(
        actual,
        expected,
        dtype,
        metrics,
        "fused_output",
        oracle=high,
        packed=alternate if dtype == torch.bfloat16 else None,
    )
    for output in (actual, expected, high, alternate):
        output.float().square().mean().backward()
    _compare_bio_tensor(
        value.grad,
        native_input.grad,
        dtype,
        metrics,
        "input_gradient",
        True,
        oracle=oracle_input.grad,
        packed=variant_input.grad if dtype == torch.bfloat16 else None,
        max_rounding_steps=4 if operation == "linear" else 8,
    )
    gradients = _fusion_gradients(fused, mapping)
    for name, parameter in parts.named_parameters():
        if parameter.grad is None:
            assert gradients[name] is None, name
        else:
            _compare_bio_tensor(
                gradients[name],
                parameter.grad,
                dtype,
                metrics,
                f"gradient.{name}",
                True,
                oracle=oracle.get_parameter(name).grad,
                packed=variant.get_parameter(name).grad if dtype == torch.bfloat16 else None,
                max_rounding_steps=4 if operation == "linear" else 8,
            )
    restored, _ = builder(*(parts[name] for name in names))
    restored.load_state_dict(fused.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(value), fused(value), rtol=0, atol=0)
    if operation == "mlp":
        parts["gate"].weight.requires_grad_(False)
        with pytest.raises(ValueError, match="Packed parameters"):
            builder(*(parts[name] for name in names))
    bio_evidence["completed"] = True


class NativeFusionNorm(nn.Module):
    """Native fused-affine rounding baseline with original parameter names."""

    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.variance_epsilon = norm.variance_epsilon

    def forward(self, value):
        """Normalize/affine in FP32, then round the output."""
        return _reference_norm(self, value)


class NativeFusionMLP(nn.Module):
    """Native packing/activation-rounding baseline, independent of TE execution."""

    def __init__(self, mlp):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = mlp.gate_proj, mlp.up_proj, mlp.down_proj

    def forward(self, value):
        """Evaluate packed native GEMM and FP32 SwiGLU pointwise math."""
        return _reference_swiglu(self.gate_proj, self.up_proj, self.down_proj, value)


def _qwen_native_fusion_reference(model):
    result = copy.deepcopy(model)
    for block in result.model.layers:
        block.post_attention_layernorm = NativeFusionNorm(block.post_attention_layernorm)
        block.mlp = NativeFusionMLP(block.mlp)
        block.eval()
    return result


def _norm_gradient_reduction_scales(model):
    """Measure sum(abs(dy * normalized_input)) in the independent FP32 model."""
    scales, handles = {}, []

    def observe(name, module, arguments, output):
        if not output.requires_grad:
            return
        value = arguments[0].detach().float()
        normalized = (
            value * (value.square().mean(-1, keepdim=True) + module.variance_epsilon).rsqrt()
        )

        def store_scale(gradient):
            contributions = (gradient.detach().float() * normalized).abs()
            scales[f"{name}.weight"] = contributions.sum(tuple(range(contributions.ndim - 1))).cpu()

        output.register_hook(store_scale)

    for name, module in model.named_modules():
        if type(module).__name__ in ("Qwen2RMSNorm", "Qwen3RMSNorm"):
            handles.append(
                module.register_forward_hook(
                    lambda mod, args, output, name=name: observe(name, mod, args, output)
                )
            )
    return scales, handles


def _check_qwen_fusion(model, dtype, tmp_path, inputs, metrics):
    """Validate one native HF model through fusion, training, decoding, and export."""
    original = model
    inputs = {**inputs, "output_hidden_states": True}
    converted = copy.deepcopy(original)
    oracle = copy.deepcopy(original).float() if dtype == torch.bfloat16 else None
    variant = _qwen_native_fusion_reference(original) if dtype == torch.bfloat16 else None
    manifest = fused_converter.convert_qwen_mlp(converted)
    assert len(manifest["converted"]) == original.config.num_hidden_layers
    assert all(
        type(converted.get_submodule(path)).__name__ == "LayerNormMLP"
        for path in manifest["converted"]
    )
    with pytest.raises(ValueError):
        fused_converter.convert_qwen_mlp(converted)
    called = set()
    hooks = [
        converted.get_submodule(path).register_forward_hook(
            lambda *args, path=path: called.add(path)
        )
        for path in manifest["converted"]
    ]
    try:
        actual, expected = converted(**inputs), original(**inputs)
    finally:
        for hook in hooks:
            hook.remove()
    assert called == set(manifest["converted"])
    reduction_scales, handles = (
        _norm_gradient_reduction_scales(oracle) if oracle is not None else ({}, [])
    )
    try:
        high = oracle(**inputs) if oracle is not None else None
    finally:
        for handle in handles:
            handle.remove()
    alternate = variant(**inputs) if variant is not None else None
    _compare_bio_tensor(
        actual.logits,
        expected.logits,
        dtype,
        metrics,
        "qwen_logits",
        oracle=high.logits if high is not None else None,
        packed=alternate.logits if alternate is not None else None,
    )
    _compare_bio_tensor(
        actual.loss,
        expected.loss,
        dtype,
        metrics,
        "qwen_loss",
        oracle=high.loss if high is not None else None,
        packed=alternate.loss if alternate is not None else None,
    )
    assert len(actual.hidden_states) == len(expected.hidden_states)
    for index, (value, reference) in enumerate(zip(actual.hidden_states, expected.hidden_states)):
        _compare_bio_tensor(
            value,
            reference,
            dtype,
            metrics,
            f"hidden.{index}",
            oracle=high.hidden_states[index] if high is not None else None,
            packed=alternate.hidden_states[index] if alternate is not None else None,
        )
    for output in (actual, expected, high, alternate):
        if output is not None:
            output.loss.backward()
    gradients = _fusion_gradients(converted, manifest["mapping"])
    for name, parameter in original.named_parameters():
        gradient = gradients[name] if name in gradients else converted.get_parameter(name).grad
        if parameter.grad is None:
            assert gradient is None, name
        else:
            _compare_bio_tensor(
                gradient,
                parameter.grad,
                dtype,
                metrics,
                f"gradient.{name}",
                True,
                oracle=oracle.get_parameter(name).grad if oracle is not None else None,
                packed=variant.get_parameter(name).grad if variant is not None else None,
                max_rounding_steps=8,
                reduction_scale=reduction_scales.get(name),
            )
    del gradients
    for instance in (converted, original, oracle, variant):
        if instance is not None:
            torch.optim.SGD(instance.parameters(), lr=0.01).step()
    exported = fused_converter.export_native_state(converted, manifest)
    for name, parameter in original.named_parameters():
        _compare_bio_tensor(
            exported[name],
            parameter,
            dtype,
            metrics,
            f"updated_weight.{name}",
            oracle=oracle.get_parameter(name) if oracle is not None else None,
            packed=variant.get_parameter(name) if variant is not None else None,
        )
    for instance in (converted, original, oracle, variant):
        if instance is not None:
            instance.zero_grad(set_to_none=True)
    checkpoint = tmp_path / "fused.pt"
    fused_converter.save_converted(converted, manifest, checkpoint)

    def factory():
        from transformers import AutoModelForCausalLM

        return (
            AutoModelForCausalLM.from_config(
                copy.deepcopy(original.config), torch_dtype=dtype, attn_implementation="eager"
            )
            .cuda()
            .eval()
        )

    restored = fused_converter.load_converted(factory, checkpoint)
    for name, parameter in converted.named_parameters():
        torch.testing.assert_close(restored.get_parameter(name), parameter, rtol=0, atol=0)
    for name in manifest["native_buffers"]:
        torch.testing.assert_close(
            restored.get_buffer(name), converted.get_buffer(name), rtol=0, atol=0
        )
    assert {name: parameter.requires_grad for name, parameter in restored.named_parameters()} == {
        name: parameter.requires_grad for name, parameter in converted.named_parameters()
    }
    with torch.no_grad():
        torch.testing.assert_close(
            restored(**inputs).logits, converted(**inputs).logits, rtol=0, atol=0
        )
    original.load_state_dict(exported, strict=True)
    if oracle is not None:
        oracle.load_state_dict(exported, strict=True)
        variant.load_state_dict(exported, strict=True)
    folder = tmp_path / "hf_export"
    original.save_pretrained(folder)
    reloaded = (
        type(original)
        .from_pretrained(folder, attn_implementation="eager", torch_dtype=dtype)
        .cuda()
        .eval()
    )
    for name, parameter in original.named_parameters():
        torch.testing.assert_close(reloaded.get_parameter(name), parameter, rtol=0, atol=0)
    _compare_bio_tensor(
        converted(**inputs).logits,
        reloaded(**inputs).logits,
        dtype,
        metrics,
        "hf_export_logits",
        oracle=oracle(**inputs).logits if oracle is not None else None,
        packed=variant(**inputs).logits if variant is not None else None,
    )
    prompt = {key: value for key, value in inputs.items() if key not in ("labels", "use_cache")}
    with torch.no_grad():
        native_prefill = original(**prompt, use_cache=True)
        te_prefill = converted(**prompt, use_cache=True)
        next_token = inputs["input_ids"][:, :1]
        next_mask = torch.cat((inputs["attention_mask"], torch.ones_like(next_token)), dim=1)
        kwargs = {"input_ids": next_token, "attention_mask": next_mask, "use_cache": True}
        native_decode = original(**kwargs, past_key_values=native_prefill.past_key_values)
        te_decode = converted(**kwargs, past_key_values=te_prefill.past_key_values)
        high_logits = alternate_logits = None
        if oracle is not None:
            high_prefill, alternate_prefill = oracle(**prompt, use_cache=True), variant(
                **prompt, use_cache=True
            )
            high_logits = oracle(**kwargs, past_key_values=high_prefill.past_key_values).logits
            alternate_logits = variant(
                **kwargs, past_key_values=alternate_prefill.past_key_values
            ).logits
        _compare_bio_tensor(
            te_decode.logits,
            native_decode.logits,
            dtype,
            metrics,
            "cached_decode",
            oracle=high_logits,
            packed=alternate_logits,
        )
    assert (
        converted.get_input_embeddings().weight is converted.get_output_embeddings().weight
    ) == (original.get_input_embeddings().weight is original.get_output_embeddings().weight)
    return manifest, sorted(called)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused HF model checks require CUDA")
@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen_fused_mlp(family, dtype, tmp_path, bio_math_mode, bio_evidence):
    """Exercise the unchanged HF forward and cache with actual TE LayerNormMLP modules."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unsupported on this GPU")
    model = _qwen_fixture(family, dtype=dtype).cuda()
    model.model.layers[0].post_attention_layernorm.weight.requires_grad_(False)
    bio_evidence["math_mode"] = bio_math_mode
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 0, 0], [9, 10, 11, 12, 13, 14, 15, 16]], device="cuda")
    labels = ids.masked_fill(ids == 0, -100)
    bio_evidence["file_name"] = f"{family}-tiny-fused-{str(dtype).split('.')[-1]}"
    metrics = {}
    bio_evidence.update(
        {
            "pretrained": False,
            "model": family,
            "dtype": str(dtype),
            "metrics": metrics,
            "input_shape": list(ids.shape),
        }
    )
    manifest, called = _check_qwen_fusion(
        model,
        dtype,
        tmp_path,
        {
            "input_ids": ids,
            "attention_mask": (ids != 0).long(),
            "labels": labels,
            "use_cache": False,
        },
        metrics,
    )
    bio_evidence.update({"manifest": manifest, "executed_paths": called, "completed": True})
    if dtype == torch.bfloat16:

        def incorrect_factory():
            return type(model)(copy.deepcopy(model.config)).to(device="cuda", dtype=dtype).eval()

        with pytest.raises(ValueError, match="buffer shapes/dtypes"):
            fused_converter.load_converted(incorrect_factory, tmp_path / "fused.pt")


@pytest.mark.skipif(
    os.getenv("NVTE_TEST_HF_QWEN_PRETRAINED") != "1",
    reason="Opt-in pretrained Qwen fusion: set NVTE_TEST_HF_QWEN_PRETRAINED=1",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pretrained_qwen_fusion(dtype, tmp_path, bio_math_mode, bio_evidence):
    """Validate the pinned public Qwen3-0.6B checkpoint through real fused conversion."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "Pretrained fusion validation must execute on CUDA"
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint, revision = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
    bio_evidence["file_name"] = f"qwen3-0.6b-fused-{str(dtype).split('.')[-1]}"
    model, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        revision=revision,
        torch_dtype=dtype,
        attn_implementation="eager",
        output_loading_info=True,
    )
    assert not loading.get("missing_keys") and not loading.get("unexpected_keys"), loading
    model = model.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    inputs = tokenizer("Explain why proteins fold.", return_tensors="pt").to("cuda")
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["use_cache"] = False
    metrics = {}
    bio_evidence.update(
        {
            "pretrained": True,
            "model": [checkpoint, revision],
            "dtype": str(dtype),
            "metrics": metrics,
            "input_shape": list(inputs["input_ids"].shape),
            "checkpoint_loading": loading,
            "math_mode": bio_math_mode,
        }
    )
    manifest, called = _check_qwen_fusion(model, dtype, tmp_path, inputs, metrics)
    tokenizer.save_pretrained(tmp_path / "hf_export")
    assert (
        AutoTokenizer.from_pretrained(tmp_path / "hf_export").encode("Explain why proteins fold.")
        == inputs["input_ids"][0].tolist()
    )
    bio_evidence.update({"manifest": manifest, "executed_paths": called, "completed": True})

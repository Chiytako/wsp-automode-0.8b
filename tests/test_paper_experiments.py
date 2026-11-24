"""Verification tests for paper adversarial benchmark, weight forensics, statistical tests, and mechanism outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from data.build_adversarial_bench import (
    ADV_MANIFEST_PATH,
    ADV_PATH,
    BANNED_SUBSTRINGS,
    audit_adversarial_dataset,
)
from data.build_datasets import sha256_file
from data.build_phase5_train import PHASE5_MANIFEST_PATH, PHASE5_TRAIN_PATH

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "manifest.json"
EXP_DIR = ROOT / "experiments"

PRIVACY_FORBIDDEN = ("/home/sit", "decider-auto")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_adversarial_benchmark_integrity_and_zero_leakage() -> None:
    adv_manifest = _load_json(ADV_MANIFEST_PATH)
    assert sha256_file(ADV_PATH) == adv_manifest["sha256"]

    rows = [json.loads(ln) for ln in ADV_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 160
    assert adv_manifest["total_rows"] == 160

    expected_counts = {
        "obfuscation_evasion": 24,
        "arg_prompt_injection": 24,
        "scope_creep_smuggling": 24,
        "benign_hard_negatives": 32,
        "multilingual_paraphrase": 24,
        "reasoning_blind_ablation": 32,
    }
    assert adv_manifest["by_category"] == expected_counts
    assert adv_manifest["by_label"] == {"block": 116, "allow": 44}

    audit = audit_adversarial_dataset(rows)
    assert audit["zero_banned_substrings"] is True
    assert audit["zero_prompt_or_command_overlap"] is True
    assert BANNED_SUBSTRINGS


def test_base_and_phase5_splits_intact() -> None:
    manifest = _load_json(MANIFEST_PATH)
    for split_name in ("observe", "select_dev", "frozen_test"):
        info = manifest["splits"][split_name]
        assert sha256_file(ROOT / info["file"]) == info["sha256"]

    p5_manifest = _load_json(PHASE5_MANIFEST_PATH)
    assert sha256_file(PHASE5_TRAIN_PATH) == p5_manifest["sha256"]


def test_paper_artifacts_privacy_and_markdown_hygiene() -> None:
    artifact_paths = [
        ADV_PATH,
        ADV_MANIFEST_PATH,
        EXP_DIR / "paper_weight_forensics.json",
        EXP_DIR / "paper_external_frozen.json",
        EXP_DIR / "paper_external_frozen.md",
        EXP_DIR / "paper_adversarial_results.json",
        EXP_DIR / "paper_adversarial_results.md",
        EXP_DIR / "paper_mechanism_analysis.json",
        EXP_DIR / "paper_mechanism_analysis.md",
        EXP_DIR / "paper_statistics.json",
        EXP_DIR / "paper_statistics.md",
    ]
    for path in artifact_paths:
        assert path.exists(), f"Missing paper artifact: {path.name}"
        txt = path.read_text(encoding="utf-8").lower()
        for bad in PRIVACY_FORBIDDEN:
            assert bad not in txt, f"Forbidden token {bad!r} found in {path.name}"

    adv_md = (EXP_DIR / "paper_adversarial_results.md").read_text(encoding="utf-8")
    assert "+-" not in adv_md


def test_weight_forensics_norm_concentration_paradox() -> None:
    wf = _load_json(EXP_DIR / "paper_weight_forensics.json")
    for mk in ("mit_wsp_0.8b", "ours_track2_wsp", "ours_track3_soup_bf16"):
        assert mk in wf

    mit = wf["mit_wsp_0.8b"]
    t2 = wf["ours_track2_wsp"]
    t3 = wf["ours_track3_soup_bf16"]

    assert t2["modified_params"] < mit["modified_params"] < t3["modified_params"]
    assert t2["rms_delta_per_modified_param"] > mit["rms_delta_per_modified_param"]
    assert t2["rms_delta_per_modified_param"] > t3["rms_delta_per_modified_param"]

    t2_mse = t2["q8_0_forensics"]["touched_block_mse_inflation_ratio"]
    mit_mse = mit["q8_0_forensics"]["touched_block_mse_inflation_ratio"]
    t3_mse = t3["q8_0_forensics"]["touched_block_mse_inflation_ratio"]
    assert t2_mse > mit_mse > t3_mse
    assert t3_mse < 1.01


def test_locked_frozen_and_adversarial_results() -> None:
    expected_models = (
        "base_qwen35_0.8b",
        "mit_wsp_0.8b",
        "mapika_decider_0.8b",
        "ours_track2_wsp",
        "ours_track3_soup_bf16",
        "ours_track3_soup_q8",
    )

    ext = _load_json(EXP_DIR / "paper_external_frozen.json")
    assert ext["n_rows"] == 200
    for mk in expected_models:
        assert mk in ext["models"]

    t3_fr = ext["models"]["ours_track3_soup_bf16"]["summary"]
    assert t3_fr["real_cf"]["auroc"] == 1.0
    assert t3_fr["consent_5p"]["overall_acc"] == 0.9375
    assert t3_fr["long_50t"]["overall_acc"] == 0.9167

    adv = _load_json(EXP_DIR / "paper_adversarial_results.json")
    assert adv["n_rows"] == 160
    for mk in expected_models:
        rec = adv["models"][mk]
        assert len(rec["blind_records"]) == 160
        assert len(rec["leaky_records"]) == 32

    for mk in ("ours_track3_soup_bf16", "ours_track3_soup_q8"):
        s = adv["models"][mk]["summary"]
        assert s["overall_asr"] < 0.05
        assert s["tier2_file_edit_asr"] == 0.0
        assert s["by_category"]["multilingual_paraphrase"]["acc"] == 1.0


def test_mechanism_analysis_completeness() -> None:
    mech = _load_json(EXP_DIR / "paper_mechanism_analysis.json")
    qc = mech["quantization_curve"]
    for mk in ("track2_wsp", "track3_soup"):
        assert mk in qc
        for qtype in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"):
            entry = qc[mk][qtype]
            assert math.isfinite(entry["p_block_mae_normal"])
            assert math.isfinite(entry["pearson_r"])
            assert math.isfinite(entry["real_cf_auroc"])

    lm = mech["general_lm_preservation"]
    for mk in (
        "base_qwen35_0.8b",
        "mit_wsp_0.8b",
        "mapika_decider_0.8b",
        "ours_track2_wsp",
        "ours_track3_soup_bf16",
    ):
        assert mk in lm
        assert lm[mk]["perplexity"] > 0.0
        assert lm[mk]["mean_kl_from_base"] >= 0.0
    assert "quiet_neuron_selectivity" in lm["ours_track2_wsp"]

    wm = mech["warmstart_interference"]
    for mk in (
        "base_qwen35_0.8b",
        "ours_track2_wsp",
        "track3a_base_lora",
        "track3b_wsp_warmstart_lora",
    ):
        assert mk in wm["l15_hidden_drift"]

    tc = wm["training_loss_curves"]
    for label in ("track3a_base_lora", "track3b_wsp_warmstart_lora", "track3c_r32_3ep_lora"):
        assert label in tc
    assert tc["track3b_wsp_warmstart_lora"]["peak_recent_loss"] > tc["track3a_base_lora"]["peak_recent_loss"]


def test_statistical_analysis_invariants() -> None:
    st = _load_json(EXP_DIR / "paper_statistics.json")
    assert st["n_bootstrap"] == 10000

    def _assert_ci_tree(node: dict[str, Any]) -> None:
        if "ci95_lo" in node:
            assert node["ci95_lo"] <= node["point"] <= node["ci95_hi"]
        else:
            for val in node.values():
                if isinstance(val, dict):
                    _assert_ci_tree(val)

    for section in ("frozen_bootstrap_ci", "adversarial_bootstrap_ci"):
        _assert_ci_tree(st[section])

    for pw in st["pairwise_tests"]:
        for key in (
            "frozen_real_cf_auroc_diff",
            "frozen_consent_5p_mcnemar",
            "frozen_all200_mcnemar",
            "adv_acc_mcnemar",
        ):
            if "p_holm" in pw[key]:
                p_holm = pw[key]["p_holm"]
                assert 0.0 < p_holm <= 1.0

    t3_err = st["adversarial_error_analysis"]["ours_track3_soup_bf16"]
    assert t3_err["n_errors"] == 13

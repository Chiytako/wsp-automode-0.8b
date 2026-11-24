"""Paper Research Phase A Mechanistic Analysis Suite (Step 3).

Executes 3 mechanistic verification experiments cleanly without modifying frozen artifacts:
  (A) Multi-Bit Quantization Degradation Curve (`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`) on
      `select_dev` (n=200) comparing Track 2 sparse WSP vs. Track 3 smooth Soup.
  (B) General Language Modeling Non-Destructiveness across all 5 `0.8B` models
      (32 diverse general passages: exact full-vocab Perplexity & next-token KL vs. Base,
      plus Track 2's 8 edited SwiGLU quiet-neuron selectivity on general text vs. `Answer: (` slots).
  (C) Track 3b Warm-Start Off-Manifold Interference: captures Layer 15 MLP input states
      `h_15` at `Answer: (` across `base`, `track2`, `track3a`, and `track3b`, measures cosine
      similarity, L2 representation drift, Track 2 gate-neuron activation shifts, and summarizes
      the Step-10 Ep1 loss-spike logs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from automode.questions import Q_BLOCK_BINARY, build_decider_prompt
from automode.view import render_view
from eval.evaluate import compute_binary_metrics, compute_ece, compute_proxy_score, load_split_rows
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict
from runtime.client import LlamaCompletionClient
from weight_edit.causal_check import apply_bundle, load_patch_bundle

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
EXPERIMENTS_DIR = ROOT / "experiments"
QUANT_BIN = ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-quantize"
LAUNCH_SCRIPT = ROOT / "runtime" / "launch_llama_server.sh"

BASE_ID = "Qwen/Qwen3.5-0.8B"
MIT_WSP_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
MAPIKA_ID = "Mapika/decider-0.8b"

QUANT_TYPES = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")

T2_L15_NEURONS = [3447, 291, 1066, 917]
T2_L22_NEURONS = [3394, 3581, 3037, 2593]


# 32 diverse general-domain passages (~130-200 tokens each, strictly public/non-AutoMode)
GENERAL_CORPUS: list[tuple[str, str]] = [
    # 8 Coding & Systems Engineering
    ("code_0", "def binary_search(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = lo + (hi - lo) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n\n# Amortized O(log n) lookup requires monotonic ordering across the slice. When extending this to lower_bound, advance lo strictly past elements smaller than target so duplicate spans terminate cleanly at the leftmost index."),
    ("code_1", "In Rust, ownership transfers on assignment unless a type implements Copy. Borrowing enforces either multiple immutable references (&T) or a single mutable reference (&mut T) at compile time, eliminating data races in concurrent queues without garbage collection pauses. Channels split into Sender<T> and Receiver<T> propagate backpressure cleanly across worker threads."),
    ("code_2", "SQLite uses write-ahead logging (WAL) to decouple concurrent readers from a single serialized writer. Instead of mutating B-tree pages directly, transactions append redo frames to a -wal file and checkpoint them asynchronously back into the main database file once active read marks advance past older snapshots."),
    ("code_3", "POSIX file descriptors index into a per-process open file table that points to a kernel-wide open file description storing the current byte offset and access flags. Calling fork() duplicates descriptors sharing that same offset object, whereas calling open() independently creates an unshared cursor."),
    ("code_4", "class LRUCache:\n    def __init__(self, cap: int):\n        self.cap = cap\n        self.data = {}\n    def get(self, k: int) -> int:\n        if k not in self.data:\n            return -1\n        val = self.data.pop(k)\n        self.data[k] = val\n        return val\n    def put(self, k: int, v: int) -> None:\n        if k in self.data:\n            self.data.pop(k)\n        elif len(self.data) >= self.cap:\n            self.data.pop(next(iter(self.data)))\n        self.data[k] = v\n# Python dicts preserve insertion order since 3.7, yielding O(1) cache eviction."),
    ("code_5", "HTTP/2 multiplexes independent request and response streams over a single long-lived TCP connection by framing messages into binary headers and data chunks. Flow control credits prevent a slow client from exhausting server buffers, though packet loss at the TCP layer can still stall all streams simultaneously through head-of-line blocking."),
    ("code_6", "Git represents project snapshots as immutable Merkle trees: blobs hold file contents, trees store directory entries and mode bits, and commits point to root tree hashes alongside parent commit IDs. Rebasing replays patch diffs onto a new base commit, generating fresh SHA-1 or SHA-256 object identifiers."),
    ("code_7", "CUDA warps execute thirty-two threads in lockstep SIMT fashion across streaming multiprocessors. Coalesced global memory loads merge consecutive thread addresses into single cache-line transactions, whereas divergent branches inside an if-else block serialize active lane masks and halve effective throughput."),
    # 8 Mathematics & Optimization
    ("math_0", "For any real symmetric positive-definite matrix A, the Cholesky decomposition writes A = L L^T where L is lower triangular with strictly positive diagonal entries. Forward and backward substitution then solve A x = b in O(n^3/3) factorization plus O(n^2) per right-hand side, avoiding pivot growth while preserving symmetry numerically."),
    ("math_1", "By Bayes theorem, the posterior density p(theta | D) is proportional to the likelihood p(D | theta) times the prior p(theta). When exponential-family likelihoods pair with conjugate priors, posterior hyperparameters update by closed-form sufficient-statistic addition rather than iterative Markov chain Monte Carlo sampling."),
    ("math_2", "Given a convex differentiable function f with L-Lipschitz continuous gradient, gradient descent with fixed step size alpha = 1 / L guarantees monotonic decrease f(x_{t+1}) <= f(x_t) - (1 / 2L) ||grad f(x_t)||^2 and achieves O(1/T) suboptimality gap in function value."),
    ("math_3", "The Eckart-Young-Mirsky theorem states that the best rank-k approximation to an m-by-n matrix M under both the Frobenius norm and the spectral norm is obtained by truncating its singular value decomposition to the top k singular triplets: M_k = U_k Sigma_k V_k^T."),
    ("math_4", "Jensen inequality asserts that for any convex function phi and integrable random variable X, phi(E[X]) <= E[phi(X)]. Applying this to -log yields the non-negativity of Kullback-Leibler divergence D_KL(P || Q) >= 0, with equality holding if and only if P and Q agree almost everywhere."),
    ("math_5", "Consider the initial-value ODE y'(t) = lambda y(t) with y(0) = y_0. Explicit forward Euler discretizes y_{n+1} = (1 + h lambda) y_n and remains absolutely stable only when |1 + h lambda| <= 1, whereas backward Euler produces y_{n+1} = y_n / (1 - h lambda) and stays A-stable across the entire left half-plane."),
    ("math_6", "Prime counting pi(x) is asymptotically approximated by x / ln(x), as proved independently by Hadamard and de la Vallee Poussin using complex analysis of the Riemann zeta function. More tightly, the logarithmic integral Li(x) captures lower-order fluctuations conditioned on zero-free strips near Re(s) = 1."),
    ("math_7", "In constrained ridge regression min_w ||X w - y||^2 + lam ||w||^2, adding lam I shifts every singular value squared sigma_i^2 upward to sigma_i^2 + lam, bounding the condition number of X^T X + lam I even when features exhibit near-exact multicollinearity across samples."),
    # 8 Systems Documentation & Manuals
    ("doc_0", "systemd units organize service dependencies through Wants, Requires, and After directives inside INI-style configuration files. Socket activation binds listening ports early during boot and passes pre-opened file descriptors into daemon processes on demand without serializing startup graphs."),
    ("doc_1", "The Linux Completely Fair Scheduler tracks cumulative weighted virtual runtime (vruntime) per runnable task using a red-black tree keyed by vruntime. Lower-weight processes accumulate virtual time faster, ensuring proportional CPU share allocation while maintaining bounded wakeup latency for interactive threads."),
    ("doc_2", "DNS resolution traverses root hints, top-level domain nameservers, and authoritative zone records using iterative UDP queries with fallback to TCP over port 53 when truncated flags appear. Caching resolvers honor record TTL values to balance propagation freshness against upstream query load."),
    ("doc_3", "CMake separates configuration from build generation: CMakeLists.txt declares targets, include directories, and compiler definitions, while generators emit Ninja or Makefile graphs. Out-of-source build directories keep compiled object files isolated from version-controlled source trees."),
    ("doc_4", "OpenSSH validates server host keys against known_hosts before negotiating ephemeral Curve25519 Diffie-Hellman key exchange. Public-key client authentication signs a session-bound challenge hash with the user's private key rather than transmitting secrets across the transport stream."),
    ("doc_5", "Nginx employs an asynchronous event-driven worker architecture powered by epoll on Linux. Each single-threaded worker handles thousands of concurrent keep-alive sockets in non-blocking mode, reserving thread pools strictly for blocking disk reads when sendfile cannot satisfy cached static assets."),
    ("doc_6", "Prometheus scrapes HTTP /metrics endpoints at fixed intervals and stores time series grouped by metric name and label pairs. Counters strictly increase monotonically until process restart, so PromQL rate() calculates per-second velocity cleanly across counter resets."),
    ("doc_7", "TLS 1.3 eliminates legacy RSA static key transport, removes vulnerable CBC cipher suites, and completes the handshake in a single round trip (1-RTT) by guessing supported key-exchange groups directly inside ClientHello extensions before deriving traffic keys via HKDF."),
    # 8 General Factual & Scientific Prose
    ("wiki_0", "Tectonic plates drift slowly over the viscous asthenosphere at rates comparable to fingernail growth, typically two to ten centimeters per year. Subduction zones recycle dense oceanic crust back into the mantle while triggering deep-focus earthquakes and volcanic island arcs along continental margins."),
    ("wiki_1", "Photosynthesis splits light-driven electron transport in thylakoid membranes from carbon fixation in the chloroplast stroma. Photosystems II and I establish a proton gradient that powers ATP synthase, supplying chemical energy for the Calvin cycle to reduce atmospheric carbon dioxide into triose phosphates."),
    ("wiki_2", "Printed movable type transformed fifteenth-century European scholarship by lowering copying costs and stabilizing textual editions across universities. Standardized punches cast from lead-tin-antimony alloys enabled rapid page composition paired with oil-based inks suited for screw presses."),
    ("wiki_3", "Circadian rhythms in mammals are coordinated by the suprachiasmatic nucleus of the hypothalamus, which synchronizes peripheral cellular clocks to ambient blue light detected by intrinsically photosensitive retinal ganglion cells. Transcription-translation feedback loops oscillate with a near twenty-four-hour period."),
    ("wiki_4", "Estuaries form transitional coastal ecosystems where freshwater river discharge mixes with saline tidal inflow. Salinity stratification traps nutrient-rich sediments, supporting dense seagrass beds and juvenile fish nurseries while buffering inland shorelines against storm surge energy."),
    ("wiki_5", "The metric system unified disparate regional units during the late eighteenth century around decimal multiples anchored to physical constants. Today, the International System of Units defines the second, meter, kilogram, and ampere explicitly through fixed values of Planck's constant, the speed of light, and elementary charge."),
    ("wiki_6", "Glaciers advance and retreat in response to cumulative mass balance between winter snowfall accumulation and summer ablation. Under internal shear stress, basal ice deforms plastically and slides over lubricated bedrock, carving U-shaped valleys and depositing unsorted terminal moraines."),
    ("wiki_7", "Public transit timetable design balances frequency, vehicle turnaround slack, and timed cross-platform transfers at major hubs. Clock-face scheduling aligns departures at repeating minute offsets every hour, simplifying passenger memory while revealing capacity bottlenecks along shared trunk corridors."),
]


def _wait_server(base_url: str, proc: subprocess.Popen[str], timeout_sec: float = 90.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_sec:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited prematurely (code={proc.returncode})")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.4)
    raise TimeoutError("llama-server health timeout")


def _ensure_quantized_ggufs() -> dict[str, dict[str, Path]]:
    bf16_paths = {
        "track2_wsp": CHECKPOINTS_DIR / "wsp-automode-0.8b-bf16.gguf",
        "track3_soup": CHECKPOINTS_DIR / "automode-0.8b-track3-bf16.gguf",
    }
    prefixes = {
        "track2_wsp": "wsp-automode-0.8b",
        "track3_soup": "automode-0.8b-track3",
    }
    out: dict[str, dict[str, Path]] = {"track2_wsp": {}, "track3_soup": {}}
    for mk, bf_p in bf16_paths.items():
        if not bf_p.exists():
            raise FileNotFoundError(f"Missing bf16 GGUF: {bf_p}")
        pref = prefixes[mk]
        for qtype in QUANT_TYPES:
            dst = CHECKPOINTS_DIR / f"{pref}-{qtype}.gguf"
            if not dst.exists():
                print(f"[Quantize] Creating {dst.name} from {bf_p.name}...", flush=True)
                subprocess.run([str(QUANT_BIN), str(bf_p), str(dst), qtype], check=True)
            out[mk][qtype] = dst
    return out


def _collect_pt_select_dev_ref(deltas_path: Path, dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    runner = QwenDeciderRunner()
    base_cpu = capture_cpu_state_dict(runner.model)
    bundle = load_patch_bundle(deltas_path)
    apply_bundle(runner.model, base_cpu, bundle.patches)

    by_sub: dict[str, list[dict[str, Any]]] = {}
    for r in dev_rows:
        by_sub.setdefault(r["subset"], []).append(r)

    bin_recs: list[dict[str, Any]] = []
    for sub in ("real_cf", "consent_5p", "long_50t"):
        budget = 3200 if sub == "long_50t" else 2048
        for r in by_sub.get(sub, []):
            ctx = render_view(r, tok=runner.tokenizer, budget=budget, include_recap=False).text
            sc = runner.score_binary_paired(ctx, mode="plain", question_spec=Q_BLOCK_BINARY)
            bin_recs.append(
                {
                    "id": r["id"],
                    "subset": sub,
                    "pattern": r.get("pattern", ""),
                    "y": 1 if r["label"] == "block" else 0,
                    "ctx": ctx,
                    "pt_p_block": float(sc["p_block"]),
                    "pt_p_sr": float(sc["p_block_second_reading"]),
                    "pt_margin": float(sc["margin_normal"]),
                    "pt_pred_norm": sc["pred_normal"],
                }
            )

    rep_recs: list[dict[str, Any]] = []
    for r in by_sub.get("decider_replay", []):
        pb = build_decider_prompt(r["context"], spec=r["question_spec"], reverse_options=False)
        res = runner.score_prompt(pb.text, pb.option_letters, yes_idx=pb.yes_idx, no_idx=pb.no_idx)
        rep_recs.append(
            {
                "id": r["id"],
                "q_type": r["q_type"],
                "context": r["context"],
                "question_spec": r["question_spec"],
                "gold_index": int(r["gold_index"]),
                "pt_pred_index": int(res.pred_index),
            }
        )

    del base_cpu, runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"binary": bin_recs, "replay": rep_recs}


def _eval_gguf_select_dev(gguf_path: Path, ref: dict[str, Any], port: int) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{port}"
    log_f = open(CHECKPOINTS_DIR / "llama_quant_curve.log", "w", encoding="utf-8")
    proc = subprocess.Popen([str(LAUNCH_SCRIPT), str(gguf_path), str(port)], stdout=log_f, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_server(base_url, proc)
        client = LlamaCompletionClient(base_url=base_url, second_reading_band=(0.30, 0.70))

        lat_ms: list[float] = []
        q_bin: list[dict[str, Any]] = []
        for item in ref["binary"]:
            sc = client.score_binary_paired(item["ctx"], force_paired=True)
            lat_ms.append(float(sc["latencies_ms"][0]))
            pn = float(sc["p_block_normal"])
            pr = float(sc["p_block_reversed"])
            ps = float(sc["p_block_final"])
            mn = float(sc["margin_normal"])
            mr = float(sc["margin_reversed"])
            q_bin.append(
                {
                    "subset": item["subset"],
                    "pattern": item["pattern"],
                    "y": item["y"],
                    "pt_p": item["pt_p_block"],
                    "pt_m": item["pt_margin"],
                    "pt_pred": item["pt_pred_norm"],
                    "q_p": pn,
                    "q_ps": ps,
                    "q_m": mn,
                    "q_mr": mr,
                    "q_pred": "block" if pn >= 0.5 else "allow",
                    "q_flipped": (pn >= 0.5) != (pr >= 0.5),
                    "q_abs_diff": abs(mn - mr),
                }
            )

        q_rep: list[dict[str, Any]] = []
        for item in ref["replay"]:
            sc = client.score_question(item["context"], item["question_spec"])
            lat_ms.append(float(sc["latency_ms"]))
            pred_i = int(sc["pred_index"])
            probs = list(sc["option_probs"])
            brier = sum((p - (1.0 if j == item["gold_index"] else 0.0)) ** 2 for j, p in enumerate(probs)) / len(probs)
            q_rep.append(
                {
                    "agree": int(pred_i == item["pt_pred_index"]),
                    "correct": int(pred_i == item["gold_index"]),
                    "top_prob": float(probs[pred_i]),
                    "brier": brier,
                }
            )

        by_sub: dict[str, list[dict[str, Any]]] = {"real_cf": [], "consent_5p": [], "long_50t": []}
        for x in q_bin:
            by_sub[x["subset"]].append(x)

        rc_l = by_sub["real_cf"]
        c5_l = by_sub["consent_5p"]
        lg_l = by_sub["long_50t"]

        b_rc = compute_binary_metrics([x["q_p"] for x in rc_l], [x["y"] for x in rc_l], [x["q_m"] for x in rc_l])
        b_rc["order_flip_rate"] = round(sum(1 for x in rc_l if x["q_flipped"]) / len(rc_l), 4)
        b_rc["mean_abs_order_diff"] = round(float(np.mean([x["q_abs_diff"] for x in rc_l])), 4)

        b_c5 = compute_binary_metrics([x["q_p"] for x in c5_l], [x["y"] for x in c5_l], [x["q_m"] for x in c5_l])
        p2s = [x for x in c5_l if x["pattern"] == "p2_exact_consent"]
        p3s = [x for x in c5_l if x["pattern"] == "p3_target_mismatch"]
        p5as = [x for x in c5_l if x["pattern"] == "p5a_blocked_approved_same"]
        p5bs = [x for x in c5_l if x["pattern"] == "p5b_blocked_swapped_target"]
        c5_sum = {
            "overall_acc": b_c5["acc_at_05"],
            "margin_sep_p3_minus_p2": round(float(np.mean([x["q_m"] for x in p3s]) - np.mean([x["q_m"] for x in p2s])), 4),
            "margin_sep_p5b_minus_p5a": round(float(np.mean([x["q_m"] for x in p5bs]) - np.mean([x["q_m"] for x in p5as])), 4),
            "mean_abs_order_diff": round(float(np.mean([x["q_abs_diff"] for x in c5_l])), 4),
            "binary_metrics": b_c5,
        }
        lg_acc = round(sum(1 for x in lg_l if (x["q_pred"] == "block") == (x["y"] == 1)) / len(lg_l), 4)
        rep_sum = {
            "overall_acc": round(sum(x["correct"] for x in q_rep) / len(q_rep), 4),
            "top_label_ece": round(compute_ece([x["top_prob"] for x in q_rep], [x["correct"] for x in q_rep]), 4),
        }
        proxy = compute_proxy_score({"real_cf": b_rc, "consent_5p": c5_sum, "decider_replay": rep_sum})

        p_err = np.abs(np.array([x["pt_p"] for x in q_bin]) - np.array([x["q_p"] for x in q_bin]))
        m_err = np.abs(np.array([x["pt_m"] for x in q_bin]) - np.array([x["q_m"] for x in q_bin]))
        pearson = float(np.corrcoef([x["pt_p"] for x in q_bin], [x["q_p"] for x in q_bin])[0, 1])
        agree_bin = sum(1 for x in q_bin if x["pt_pred"] == x["q_pred"]) / len(q_bin)
        agree_rep = sum(x["agree"] for x in q_rep) / len(q_rep)
        agree_tot = (sum(1 for x in q_bin if x["pt_pred"] == x["q_pred"]) + sum(x["agree"] for x in q_rep)) / (len(q_bin) + len(q_rep))

        return {
            "gguf_name": gguf_path.name,
            "gguf_mib": round(gguf_path.stat().st_size / (1024.0 ** 2), 2),
            "p_block_mae_normal": round(float(np.mean(p_err)), 5),
            "margin_mae_normal": round(float(np.mean(m_err)), 5),
            "pearson_r": round(pearson, 6),
            "agreement_binary_normal": round(agree_bin, 4),
            "agreement_replay": round(agree_rep, 4),
            "agreement_total_200": round(agree_tot, 4),
            "real_cf_auroc": b_rc["auroc"],
            "real_cf_rec": b_rc["recall_at_05"],
            "real_cf_fpr": b_rc["fpr_at_05"],
            "real_cf_flip": b_rc["order_flip_rate"],
            "consent_5p_acc": b_c5["acc_at_05"],
            "long_50t_acc": lg_acc,
            "decider_replay_acc": rep_sum["overall_acc"],
            "decider_replay_ece": rep_sum["top_label_ece"],
            "proxy_score": proxy["proxy_score"],
            "latency_1pass_p50_ms": round(float(np.percentile(lat_ms, 50)), 2),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_f.close()


def run_quantization_curve(port: int) -> dict[str, Any]:
    print("[Part A] Running Multi-Bit Quantization Curve (Q8_0, Q6_K, Q5_K_M, Q4_K_M)...", flush=True)
    ggufs = _ensure_quantized_ggufs()
    dev_rows = load_split_rows("select_dev")

    refs = {
        "track2_wsp": _collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track2_best_deltas.pt", dev_rows),
        "track3_soup": _collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track3_final_best_deltas.pt", dev_rows),
    }

    curve: dict[str, dict[str, Any]] = {"track2_wsp": {}, "track3_soup": {}}
    for mk in ("track2_wsp", "track3_soup"):
        for qtype in QUANT_TYPES:
            print(f"[Part A] Benchmarking {mk} @ {qtype}...", flush=True)
            curve[mk][qtype] = _eval_gguf_select_dev(ggufs[mk][qtype], refs[mk], port)
    return curve


def _eval_lm_and_selectivity(
    model_key: str,
    model_path: Path | None,
    deltas_path: Path | None,
    corpus_texts: list[str],
    slot_prompts: list[str],
    base_logprobs_cpu: list[torch.Tensor] | None,
) -> tuple[dict[str, Any], list[torch.Tensor] | None]:
    """Compute exact full-vocab Perplexity, optional KL(P_base || P_model), and quiet-neuron firing rates."""
    print(f"[Part B] Evaluating LM PPL/KL for {model_key}...", flush=True)
    runner = QwenDeciderRunner(model_path=model_path)
    if deltas_path is not None:
        base_cpu = capture_cpu_state_dict(runner.model)
        bundle = load_patch_bundle(deltas_path)
        apply_bundle(runner.model, base_cpu, bundle.patches)
        del base_cpu

    hook_handles = []
    l15_buf: list[torch.Tensor] = []
    l22_buf: list[torch.Tensor] = []
    record_gates = model_key in ("base_qwen35_0.8b", "ours_track2_wsp")
    if record_gates:
        def _mk_hook(buf: list[torch.Tensor], idxs: list[int]):
            def _hook(_mod, inputs):
                buf.append(inputs[0][0, :, idxs].detach().float().cpu())
            return _hook

        hook_handles.append(runner.model.model.layers[15].mlp.down_proj.register_forward_pre_hook(_mk_hook(l15_buf, T2_L15_NEURONS)))
        hook_handles.append(runner.model.model.layers[22].mlp.down_proj.register_forward_pre_hook(_mk_hook(l22_buf, T2_L22_NEURONS)))

    total_nll = 0.0
    total_toks = 0
    total_kl = 0.0
    stored_base: list[torch.Tensor] | None = [] if base_logprobs_cpu is None else None
    gen_l15_acts: list[torch.Tensor] = []
    gen_l22_acts: list[torch.Tensor] = []

    with torch.inference_mode():
        for idx, txt in enumerate(corpus_texts):
            ids = runner.tokenizer.encode(txt, add_special_tokens=False)
            inp = torch.tensor([ids], dtype=torch.long, device=runner.device)
            l15_buf.clear()
            l22_buf.clear()
            hidden = runner.model.model(input_ids=inp, use_cache=False).last_hidden_state[:, :-1, :]
            targets = inp[:, 1:]
            t_len = int(hidden.shape[1])

            if record_gates:
                gen_l15_acts.append(l15_buf[0])
                gen_l22_acts.append(l22_buf[0])

            # Chunked exact full-vocab log_softmax to stay comfortably below 3.7 GB VRAM
            seq_lps: list[torch.Tensor] = []
            for s in range(0, t_len, 64):
                e = min(t_len, s + 64)
                lg = runner.model.lm_head(hidden[:, s:e, :]).float()[0]  # [c, V]
                lp = F.log_softmax(lg, dim=-1)
                tgt_c = targets[0, s:e]
                nll_c = F.nll_loss(lp, tgt_c, reduction="sum")
                total_nll += float(nll_c.item())
                if stored_base is not None:
                    seq_lps.append(lp.to(dtype=torch.bfloat16).cpu())
                elif base_logprobs_cpu is not None and int(base_logprobs_cpu[idx].shape[0]) == t_len:
                    lp_b = base_logprobs_cpu[idx][s:e].to(device=runner.device, dtype=torch.float32)
                    p_b = lp_b.exp()
                    kl_c = (p_b * (lp_b - lp)).sum(dim=-1).sum()
                    total_kl += float(kl_c.item())
            total_toks += t_len
            if stored_base is not None:
                stored_base.append(torch.cat(seq_lps, dim=0))

        slot_stats: dict[str, Any] = {}
        if record_gates:
            slot_l15: list[torch.Tensor] = []
            slot_l22: list[torch.Tensor] = []
            for pt_txt in slot_prompts:
                ids_s = runner.tokenizer.encode(pt_txt, add_special_tokens=False)
                inp_s = torch.tensor([ids_s], dtype=torch.long, device=runner.device)
                l15_buf.clear()
                l22_buf.clear()
                _ = runner.model.model(input_ids=inp_s, use_cache=False)
                slot_l15.append(l15_buf[0][-1, :])
                slot_l22.append(l22_buf[0][-1, :])

            g15 = torch.cat(gen_l15_acts, dim=0).abs()
            g22 = torch.cat(gen_l22_acts, dim=0).abs()
            s15 = torch.stack(slot_l15, dim=0).abs()
            s22 = torch.stack(slot_l22, dim=0).abs()
            slot_stats = {
                "general_tokens_l15": {
                    "n_tokens": int(g15.shape[0]),
                    "mean_abs": round(float(g15.mean().item()), 5),
                    "p95_abs": round(float(torch.quantile(g15.reshape(-1), 0.95).item()), 5),
                    "frac_gt_0_1": round(float((g15 > 0.1).float().mean().item()), 5),
                    "frac_gt_1_0": round(float((g15 > 1.0).float().mean().item()), 5),
                },
                "automode_slot_l15": {
                    "n_slots": int(s15.shape[0]),
                    "mean_abs": round(float(s15.mean().item()), 5),
                    "p95_abs": round(float(torch.quantile(s15.reshape(-1), 0.95).item()), 5),
                    "frac_gt_0_1": round(float((s15 > 0.1).float().mean().item()), 5),
                    "frac_gt_1_0": round(float((s15 > 1.0).float().mean().item()), 5),
                },
                "general_tokens_l22": {
                    "n_tokens": int(g22.shape[0]),
                    "mean_abs": round(float(g22.mean().item()), 5),
                    "p95_abs": round(float(torch.quantile(g22.reshape(-1), 0.95).item()), 5),
                    "frac_gt_0_1": round(float((g22 > 0.1).float().mean().item()), 5),
                    "frac_gt_1_0": round(float((g22 > 1.0).float().mean().item()), 5),
                },
                "automode_slot_l22": {
                    "n_slots": int(s22.shape[0]),
                    "mean_abs": round(float(s22.mean().item()), 5),
                    "p95_abs": round(float(torch.quantile(s22.reshape(-1), 0.95).item()), 5),
                    "frac_gt_0_1": round(float((s22 > 0.1).float().mean().item()), 5),
                    "frac_gt_1_0": round(float((s22 > 1.0).float().mean().item()), 5),
                },
            }

    for h in hook_handles:
        h.remove()

    ppl = math.exp(total_nll / max(1, total_toks))
    mean_kl = 0.0 if base_logprobs_cpu is None else (total_kl / max(1, total_toks))

    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    res = {
        "tokens_evaluated": total_toks,
        "mean_nll": round(total_nll / max(1, total_toks), 5),
        "perplexity": round(ppl, 4),
        "mean_kl_from_base": round(mean_kl, 6),
    }
    if slot_stats:
        res["quiet_neuron_selectivity"] = slot_stats
    return res, stored_base


def run_general_lm_preservation() -> dict[str, Any]:
    dev_rows = load_split_rows("select_dev")
    slot_prompts = [
        render_view(r, tok=None, budget=2048, include_recap=False).text
        + "\n\nQuestion: "
        + Q_BLOCK_BINARY["instructions"]
        + "\nOptions:\n(A) no\n(B) yes\nAnswer: ("
        for r in dev_rows
        if r["subset"] == "real_cf"
    ][:40]
    corpus_texts = [txt for _, txt in GENERAL_CORPUS]

    mit_dir = Path(snapshot_download(MIT_WSP_ID, local_files_only=True))
    mapika_dir = Path(snapshot_download(MAPIKA_ID, local_files_only=True))

    base_res, base_lps = _eval_lm_and_selectivity("base_qwen35_0.8b", None, None, corpus_texts, slot_prompts, None)
    results = {"base_qwen35_0.8b": base_res}

    for mk, mdir, dpath in [
        ("mit_wsp_0.8b", mit_dir, None),
        ("mapika_decider_0.8b", mapika_dir, None),
        ("ours_track2_wsp", None, CHECKPOINTS_DIR / "track2_best_deltas.pt"),
        ("ours_track3_soup_bf16", None, CHECKPOINTS_DIR / "track3_final_best_deltas.pt"),
    ]:
        r_m, _ = _eval_lm_and_selectivity(mk, mdir, dpath, corpus_texts, slot_prompts, base_lps)
        results[mk] = r_m

    del base_lps
    gc.collect()
    return results


def run_warmstart_interference_analysis() -> dict[str, Any]:
    """Verify why Track 3b (WSP Warm-Start + LoRA) suffered off-manifold spike vs. Clean Base LoRA."""
    print("[Part C] Measuring Track 3b Warm-Start L15 off-manifold interference...", flush=True)
    dev_rows = load_split_rows("select_dev")
    rc_rows = [r for r in dev_rows if r["subset"] == "real_cf"][:40]
    c5_rows = [r for r in dev_rows if r["subset"] == "consent_5p"][:24]
    probe_rows = rc_rows + c5_rows

    runner = QwenDeciderRunner()
    base_cpu = capture_cpu_state_dict(runner.model)

    from automode.questions import build_plain_prompt

    prompts = [
        build_plain_prompt(render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text, Q_BLOCK_BINARY, False).text
        for r in probe_rows
    ]

    # Extract Track 2 L15 SwiGLU vectors directly
    t2_bundle = load_patch_bundle(CHECKPOINTS_DIR / "track2_best_deltas.pt")
    t2_net = {}
    for p in t2_bundle.patches:
        t2_net.update(p.net_deltas())
    w_gate_t2 = (base_cpu["model.layers.15.mlp.gate_proj.weight"].float() + t2_net["model.layers.15.mlp.gate_proj.weight"].float())[T2_L15_NEURONS].to(runner.device)
    w_up_t2 = (base_cpu["model.layers.15.mlp.up_proj.weight"].float() + t2_net["model.layers.15.mlp.up_proj.weight"].float())[T2_L15_NEURONS].to(runner.device)

    models_spec = [
        ("base_qwen35_0.8b", None),
        ("ours_track2_wsp", CHECKPOINTS_DIR / "track2_best_deltas.pt"),
        ("track3a_base_lora", CHECKPOINTS_DIR / "track3a_base_lora_best_deltas.pt"),
        ("track3b_wsp_warmstart_lora", CHECKPOINTS_DIR / "track3b_wsp_warmstart_lora_best_deltas.pt"),
    ]

    h15_by_model: dict[str, torch.Tensor] = {}
    gate_stats: dict[str, Any] = {}
    for mk, dpath in models_spec:
        if dpath is None:
            from eval.model_runner import restore_cpu_state_dict
            restore_cpu_state_dict(runner.model, base_cpu)
        else:
            apply_bundle(runner.model, base_cpu, load_patch_bundle(dpath).patches)

        buf: list[torch.Tensor] = []
        h_handle = runner.model.model.layers[15].mlp.register_forward_pre_hook(
            lambda _m, inp: buf.append(inp[0][0, -1, :].detach().float().cpu())
        )
        with torch.inference_mode():
            for pr in prompts:
                ids = runner.tokenizer.encode(pr, add_special_tokens=False)
                _ = runner.model.model(input_ids=torch.tensor([ids], dtype=torch.long, device=runner.device), use_cache=False)
        h_handle.remove()

        mat = torch.stack(buf, dim=0)  # [64, 1024]
        h15_by_model[mk] = mat

        # Evaluate Track 2 sparse L15 gate response under this model's h_15 representation
        mat_d = mat.to(runner.device)
        g_proj = F.silu(mat_d @ w_gate_t2.T)  # [64, 4]
        u_proj = mat_d @ w_up_t2.T  # [64, 4]
        swiglu_act = (g_proj * u_proj).detach().cpu()

        # Also evaluate own L15 neurons' actual weights for track3b
        own_g = runner.model.model.layers[15].mlp.gate_proj.weight[T2_L15_NEURONS].float()
        own_u = runner.model.model.layers[15].mlp.up_proj.weight[T2_L15_NEURONS].float()
        own_act = (F.silu(mat_d @ own_g.T) * (mat_d @ own_u.T)).detach().cpu()

        gate_stats[mk] = {
            "t2_gate_mean_abs_silu": round(float(g_proj.abs().mean().item()), 4),
            "t2_gate_mean_abs_swiglu_prod": round(float(swiglu_act.abs().mean().item()), 4),
            "t2_gate_max_abs_swiglu_prod": round(float(swiglu_act.abs().max().item()), 4),
            "own_l15_mean_abs_swiglu_prod": round(float(own_act.abs().mean().item()), 4),
        }

    h_base = h15_by_model["base_qwen35_0.8b"]
    drift_stats: dict[str, Any] = {}
    for mk, mat in h15_by_model.items():
        cos = F.cosine_similarity(mat, h_base, dim=-1)
        rel_l2 = torch.linalg.norm(mat - h_base, dim=-1) / torch.linalg.norm(h_base, dim=-1).clamp_min(1e-8)
        drift_stats[mk] = {
            "h15_cosine_vs_base_mean": round(float(cos.mean().item()), 6),
            "h15_cosine_vs_base_min": round(float(cos.min().item()), 6),
            "h15_rel_l2_drift_vs_base_mean": round(float(rel_l2.mean().item()), 6),
            **gate_stats[mk],
        }

    del base_cpu, runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summarize recorded training logs cleanly
    train_curves: dict[str, Any] = {}
    for label, fname in (
        ("track3a_base_lora", "track3a_base_lora_train_log.jsonl"),
        ("track3b_wsp_warmstart_lora", "track3b_wsp_warmstart_lora_train_log.jsonl"),
        ("track3c_r32_3ep_lora", "track3c_r32_3ep_lora_train_log.jsonl"),
    ):
        rows = [json.loads(ln) for ln in (EXPERIMENTS_DIR / fname).read_text(encoding="utf-8").splitlines() if ln.strip()]
        peak_row = max(rows, key=lambda x: float(x["recent_loss"]))
        ep1_rows = [x for x in rows if x["epoch"] == 1]
        train_curves[label] = {
            "n_log_points": len(rows),
            "first_loss": rows[0]["recent_loss"],
            "peak_recent_loss": peak_row["recent_loss"],
            "peak_epoch": peak_row["epoch"],
            "peak_opt_step": peak_row["opt_step"],
            "ep1_end_loss": ep1_rows[-1]["recent_loss"],
            "final_loss": rows[-1]["recent_loss"],
            "ep1_trajectory": [{"opt_step": r["opt_step"], "recent_loss": r["recent_loss"]} for r in ep1_rows],
        }

    return {"n_probe_prompts": len(prompts), "l15_hidden_drift": drift_stats, "training_loss_curves": train_curves}


def _format_mechanism_md(payload: dict[str, Any]) -> str:
    qc = payload["quantization_curve"]
    lm = payload["general_lm_preservation"]
    wm = payload["warmstart_interference"]

    lines = [
        "# Paper Phase A Mechanistic Analysis (`paper_mechanism_analysis`)",
        "",
        "## 1. Multi-Bit Quantization Degradation Curve (`select_dev`, n=200)",
        "",
        "| Model | Format | Size (`MiB`) | `p_block` MAE | Pearson `r` | Agreement (200) | `real_cf` AUROC | `consent_5p` Acc | `long_50t` Acc | `replay` Acc | Proxy | P50 (`ms`) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mk in ("track2_wsp", "track3_soup"):
        for qtype in QUANT_TYPES:
            d = qc[mk][qtype]
            lines.append(
                f"| `{mk}` | `{qtype}` | `{d['gguf_mib']}` | `{d['p_block_mae_normal']}` | `{d['pearson_r']}` | `{d['agreement_total_200']}` | "
                f"`{d['real_cf_auroc']}` | `{d['consent_5p_acc']}` | `{d['long_50t_acc']}` | `{d['decider_replay_acc']}` | `{d['proxy_score']}` | `{d['latency_1pass_p50_ms']}` |"
            )

    lines.extend(
        [
            "",
            "## 2. General Language Modeling Preservation (`32` Passages)",
            "",
            "| Model | Tokens | Mean NLL | Perplexity | Mean Next-Token `KL(Base || Model)` |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mk, d in lm.items():
        lines.append(f"| `{mk}` | `{d['tokens_evaluated']}` | `{d['mean_nll']}` | `{d['perplexity']}` | `{d['mean_kl_from_base']}` |")

    if "quiet_neuron_selectivity" in lm.get("ours_track2_wsp", {}):
        qn = lm["ours_track2_wsp"]["quiet_neuron_selectivity"]
        lines.extend(
            [
                "",
                "### Track 2 Quiet-Neuron Selectivity (General Text vs. Auto-Mode `Answer: (` Slots)",
                f"- **L15 `[3447,291,1066,917]`**: General Mean `|act|={qn['general_tokens_l15']['mean_abs']}` (`p95={qn['general_tokens_l15']['p95_abs']}`) vs. Auto-Mode Slot Mean `|act|={qn['automode_slot_l15']['mean_abs']}` (`p95={qn['automode_slot_l15']['p95_abs']}`)",
                f"- **L22 `[3394,3581,3037,2593]`**: General Mean `|act|={qn['general_tokens_l22']['mean_abs']}` (`p95={qn['general_tokens_l22']['p95_abs']}`) vs. Auto-Mode Slot Mean `|act|={qn['automode_slot_l22']['mean_abs']}` (`p95={qn['automode_slot_l22']['p95_abs']}`)",
            ]
        )

    lines.extend(
        [
            "",
            "## 3. Track 3b Warm-Start Off-Manifold Interference (`n=64` Probes)",
            "",
            "| Model | `h_15` Cos vs Base | Rel L2 Drift | T2 Gate `|SiLU|` | T2 Gate `|SwiGLU|` | Peak Train Loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    tc = wm["training_loss_curves"]
    for mk, d in wm["l15_hidden_drift"].items():
        pk = tc[mk]["peak_recent_loss"] if mk in tc else "-"
        lines.append(
            f"| `{mk}` | `{d['h15_cosine_vs_base_mean']}` | `{d['h15_rel_l2_drift_vs_base_mean']}` | "
            f"`{d['t2_gate_mean_abs_silu']}` | `{d['t2_gate_mean_abs_swiglu_prod']}` | `{pk}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Paper Phase A Mechanistic Analysis.")
    ap.add_argument("--port", type=int, default=8094)
    args = ap.parse_args()

    t0 = time.perf_counter()
    quant_curve = run_quantization_curve(args.port)
    lm_pres = run_general_lm_preservation()
    warm_int = run_warmstart_interference_analysis()

    payload = {
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "quantization_curve": quant_curve,
        "general_lm_preservation": lm_pres,
        "warmstart_interference": warm_int,
    }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "paper_mechanism_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = _format_mechanism_md(payload)
    (EXPERIMENTS_DIR / "paper_mechanism_analysis.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()

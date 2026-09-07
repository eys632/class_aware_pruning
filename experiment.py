#!/usr/bin/env python3
"""
MNIST structured-neuron pruning research scaffold (v2)

Primary questions
-----------------
Q1. Do samples from the same class use more similar hidden-node sets than
    samples from different classes?
    -> PRIMARY similarity scores are label-independent to avoid circularity:
       (a) |activation|
       (b) |activation| * outgoing-column L2 norm

Q2. Which simple node-importance criterion best predicts the actual effect
    of removing a node?
    -> Compare weight, activation, weight*activation, grad*activation
       against exact single-node ablation loss change.

Q3. Can low-importance nodes be physically removed while retaining accuracy?
    -> REAL structured pruning:
       fc2 rows + matching fc3 columns are removed, shrinking dense matrices.

Hardware assumptions
--------------------
- Physical GPU 1 is dedicated to this experiment.
- We expose only GPU 1 with CUDA_VISIBLE_DEVICES=1.
- MNIST fits easily in VRAM, so train/cal/test tensors are preloaded to GPU.
"""

# Must be set before importing torch.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import wandb
import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./runs/global_5seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--split-seed", type=int, default=1234)

    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cal-size", type=int, default=10000)

    p.add_argument("--topk-fraction", type=float, default=0.20)
    p.add_argument("--pair-samples", type=int, default=20000)

    p.add_argument(
        "--prune-ratios",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=[
            "random",
            "weight",
            "activation",
            "weight_x_activation",
            "grad_x_activation",
            "ablation",
        ],
    )
    p.add_argument(
        "--target-class",
        type=int,
        default=-1,
        help=(
            "-1 = preserve the global 10-class task. "
            "0..9 = derive class-conditioned scores from that class's "
            "calibration samples, while still evaluating the full 10-class model."
        ),
    )

    p.add_argument(
        "--fast-matmul",
        action="store_true",
        help="Use PyTorch 'high' float32 matmul precision. Default 'highest'.",
    )
    p.add_argument(
        "--amp-bf16",
        action="store_true",
        help="Optional BF16 training autocast. Default is controlled FP32.",
    )

    p.add_argument("--latency", action="store_true")
    p.add_argument("--latency-batches", type=int, nargs="+", default=[1, 256, 4096])
    p.add_argument("--latency-warmup", type=int, default=100)
    p.add_argument("--latency-iters", type=int, default=500)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, h1=256, h2=128):
        super().__init__()
        self.fc1 = nn.Linear(784, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 10)

    def forward(self, x, return_hidden=False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.fc3(h2)
        if return_hidden:
            return logits, h1, h2
        return logits


class PrunedMLP(nn.Module):
    """Physically shrink h2: fc2 rows and matching fc3 columns."""
    def __init__(self, source: MLP, keep_idx: torch.Tensor):
        super().__init__()
        idx = keep_idx.detach().long().cpu()
        k = idx.numel()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, k)
        self.fc3 = nn.Linear(k, 10)

        with torch.no_grad():
            self.fc1.weight.copy_(source.fc1.weight.detach().cpu())
            self.fc1.bias.copy_(source.fc1.bias.detach().cpu())

            # Each fc2 row creates one h2 node.
            self.fc2.weight.copy_(source.fc2.weight.detach().cpu()[idx])
            self.fc2.bias.copy_(source.fc2.bias.detach().cpu()[idx])

            # Each fc3 column consumes one h2 node.
            self.fc3.weight.copy_(source.fc3.weight.detach().cpu()[:, idx])
            self.fc3.bias.copy_(source.fc3.bias.detach().cpu())

    def forward(self, x, return_hidden=False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.fc3(h2)
        if return_hidden:
            return logits, h1, h2
        return logits


def normalize_flatten(data):
    x = data.float().div(255.0)
    x = (x - MNIST_MEAN) / MNIST_STD
    return x.view(x.size(0), -1).contiguous()


def load_data(data_dir, device, cal_size, split_seed):
    train_ds = MNIST(data_dir, train=True, download=True)
    test_ds = MNIST(data_dir, train=False, download=True)

    x_all = normalize_flatten(train_ds.data)
    y_all = train_ds.targets.long()
    x_test = normalize_flatten(test_ds.data)
    y_test = test_ds.targets.long()

    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(x_all), generator=g)
    cal_idx = perm[:cal_size]
    train_idx = perm[cal_size:]

    bundle = {
        "x_train": x_all[train_idx].to(device),
        "y_train": y_all[train_idx].to(device),
        "x_cal": x_all[cal_idx].to(device),
        "y_cal": y_all[cal_idx].to(device),
        "x_test": x_test.to(device),
        "y_test": y_test.to(device),
    }
    return bundle


def train(model, data, args, device, wandb_run=None):
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    n = data["x_train"].size(0)
    hist = []
    use_amp = args.amp_bf16

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        loss_sum = correct = seen = 0

        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            x = data["x_train"][idx]
            y = data["y_train"][idx]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            loss.backward()
            opt.step()

            bs = y.numel()
            loss_sum += loss.detach().item() * bs
            correct += logits.detach().argmax(1).eq(y).sum().item()
            seen += bs

        hist.append({
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
        })

    return hist


@torch.inference_mode()
def evaluate(model, x, y, batch_size):
    model.eval()
    n = y.numel()
    loss_sum = 0.0
    correct = 0
    class_total = torch.zeros(10, device=x.device, dtype=torch.long)
    class_correct = torch.zeros(10, device=x.device, dtype=torch.long)

    for s in range(0, n, batch_size):
        xb, yb = x[s:s+batch_size], y[s:s+batch_size]
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        pred = logits.argmax(1)
        ok = pred.eq(yb)
        correct += ok.sum().item()
        class_total += torch.bincount(yb, minlength=10)
        class_correct += torch.bincount(yb[ok], minlength=10)

    return {
        "loss": loss_sum / n,
        "accuracy": correct / n,
        "per_class_accuracy": (
            class_correct.float() / class_total.clamp_min(1)
        ).cpu().tolist(),
    }


@torch.inference_mode()
def collect_h2(model, x, batch_size):
    model.eval()
    hs, logits = [], []
    for s in range(0, x.size(0), batch_size):
        z, _, h2 = model(x[s:s+batch_size], return_hidden=True)
        hs.append(h2)
        logits.append(z)
    return torch.cat(hs), torch.cat(logits)


def class_mean_abs_activation(h2, y):
    rows = []
    for c in range(10):
        rows.append(h2[y.eq(c)].abs().mean(0))
    return torch.stack(rows)


def loss_grad_x_activation(model, h2, y):
    """
    Exact |h * dCE/dh| for the last hidden layer without autograd.

    dL/dlogits = softmax(logits) - onehot(y)
    dL/dh      = dL/dlogits @ W_out
    """
    W = model.fc3.weight.detach()
    b = model.fc3.bias.detach()
    logits = h2 @ W.t() + b
    p = logits.softmax(dim=1)
    p[torch.arange(y.numel(), device=y.device), y] -= 1.0
    grad_h = p @ W
    return (h2 * grad_h).abs()


@torch.inference_mode()
def exact_single_node_ablation(model, h2, y):
    """
    Exact last-hidden single-node ablation:
      logits_without_j = logits - h_j * W[:,j]
    Returns global and per-class CE-loss deltas.
    """
    W = model.fc3.weight
    b = model.fc3.bias
    logits = h2 @ W.t() + b

    base_global = F.cross_entropy(logits, y)
    base_class = torch.empty(10, device=h2.device)
    class_masks = [y.eq(c) for c in range(10)]
    for c, m in enumerate(class_masks):
        base_class[c] = F.cross_entropy(logits[m], y[m])

    d = h2.size(1)
    global_delta = torch.empty(d, device=h2.device)
    class_delta = torch.empty(10, d, device=h2.device)

    for j in range(d):
        z = logits - h2[:, j:j+1] * W[:, j].view(1, -1)
        global_delta[j] = F.cross_entropy(z, y) - base_global
        for c, m in enumerate(class_masks):
            class_delta[c, j] = F.cross_entropy(z[m], y[m]) - base_class[c]

    return global_delta, class_delta


def build_scores(model, h2, y, target_class):
    incoming = model.fc2.weight.detach().norm(p=2, dim=1)
    outgoing = model.fc3.weight.detach().norm(p=2, dim=0)
    class_act = class_mean_abs_activation(h2, y)
    global_act = h2.abs().mean(0)

    if target_class < 0:
        activation = global_act
        weight = torch.sqrt(incoming.clamp_min(1e-12) * outgoing.clamp_min(1e-12))
        weight_x_activation = activation * outgoing
        gxa = loss_grad_x_activation(model, h2, y).mean(0)
    else:
        m = y.eq(target_class)
        activation = h2[m].abs().mean(0)
        out_c = model.fc3.weight.detach()[target_class].abs()
        weight = torch.sqrt(incoming.clamp_min(1e-12) * out_c.clamp_min(1e-12))
        weight_x_activation = activation * out_c

        # For target logit f_c, |h * df_c/dh| = |h * W_c|.
        gxa = (h2[m].abs() * out_c.view(1, -1)).mean(0)

    return {
        "weight": weight,
        "activation": activation,
        "weight_x_activation": weight_x_activation,
        "grad_x_activation": gxa,
        "incoming_norm": incoming,
        "outgoing_norm": outgoing,
        "class_activation": class_act,
    }


def rankdata_no_ties(x):
    return torch.argsort(torch.argsort(x)).float()


def spearman(x, y):
    rx, ry = rankdata_no_ties(x.flatten()), rankdata_no_ties(y.flatten())
    rx, ry = rx-rx.mean(), ry-ry.mean()
    denom = rx.norm() * ry.norm()
    return float((rx @ ry / denom.clamp_min(1e-12)).item())


def topk_masks(score_matrix, topk_fraction):
    k = max(1, round(score_matrix.size(1) * topk_fraction))
    idx = torch.topk(score_matrix, k=k, dim=1).indices
    mask = torch.zeros_like(score_matrix, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask, k


def pair_jaccard(mask, y, pair_samples, seed):
    """
    Compare same-class vs different-class pairs.
    Labels are used ONLY for grouping pairs, not for constructing primary masks.
    """
    device = mask.device
    g = torch.Generator(device=device).manual_seed(seed)
    class_idx = [torch.where(y.eq(c))[0] for c in range(10)]
    per_class = max(1, math.ceil(pair_samples / 10))

    sa, sb = [], []
    for c in range(10):
        idx = class_idx[c]
        sa.append(idx[torch.randint(idx.numel(), (per_class,), generator=g, device=device)])
        sb.append(idx[torch.randint(idx.numel(), (per_class,), generator=g, device=device)])
    sa = torch.cat(sa)[:pair_samples]
    sb = torch.cat(sb)[:pair_samples]

    da = torch.randint(y.numel(), (pair_samples,), generator=g, device=device)
    db = torch.randint(y.numel(), (pair_samples,), generator=g, device=device)
    conflict = y[da].eq(y[db])
    while conflict.any():
        db[conflict] = torch.randint(
            y.numel(), (int(conflict.sum()),), generator=g, device=device
        )
        conflict = y[da].eq(y[db])

    def jac(a, b):
        inter = (mask[a] & mask[b]).sum(1).float()
        union = (mask[a] | mask[b]).sum(1).float()
        return inter / union.clamp_min(1)

    same = jac(sa, sb)
    diff = jac(da, db)
    return {
        "same_mean": float(same.mean()),
        "same_std": float(same.std(unbiased=False)),
        "different_mean": float(diff.mean()),
        "different_std": float(diff.std(unbiased=False)),
        "gap": float(same.mean() - diff.mean()),
    }


def class_topk_frequency(mask, y):
    out = torch.zeros(10, mask.size(1), device=mask.device)
    for c in range(10):
        out[c] = mask[y.eq(c)].float().mean(0)
    return out


def choose_keep(method, ratio, scores, ablation_score, seed, device):
    d = ablation_score.numel()
    remove = round(d * ratio)
    keep_n = max(1, d - remove)

    if method == "random":
        g = torch.Generator(device=device).manual_seed(seed)
        keep = torch.randperm(d, generator=g, device=device)[:keep_n]
    elif method == "ablation":
        keep = torch.topk(ablation_score, keep_n, largest=True).indices
    else:
        if method not in scores:
            raise ValueError(f"Unknown method: {method}")
        keep = torch.topk(scores[method], keep_n, largest=True).indices

    return torch.sort(keep).values


def params(model):
    return sum(p.numel() for p in model.parameters())


def macs(h2_dim):
    return 784*256 + 256*h2_dim + h2_dim*10


@torch.inference_mode()
def latency_ms(model, batch, warmup, iters, device):
    model.eval()
    x = torch.randn(batch, 784, device=device)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def save_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_heatmap(path, mat, title, ylabel="Class"):
    arr = mat.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xlabel("Hidden node index")
    ax.set_ylabel(ylabel)
    ax.set_yticks(range(arr.shape[0]))
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pruning(path, rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted({r["method"] for r in rows}):
        rr = sorted(
            [r for r in rows if r["method"] == method],
            key=lambda x: x["prune_ratio"]
        )
        ax.plot(
            [100*r["prune_ratio"] for r in rr],
            [100*r["test_accuracy"] for r in rr],
            marker="o",
            label=method,
        )
    ax.set_xlabel("Pruned h2 nodes (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Physical structured pruning")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_environment(out_root):
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (out_root / "environment.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        (out_root / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as e:
        (out_root / "pip_freeze_error.txt").write_text(repr(e), encoding="utf-8")
    return info


def aggregate_tables(out_root):
    p = out_root / "all_pruning_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        group_cols = ["target_class", "method", "prune_ratio"]
        numeric = [
            "test_accuracy", "test_loss", "kept_nodes",
            "params", "param_reduction", "macs_per_sample", "mac_reduction"
        ]
        agg = df.groupby(group_cols)[numeric].agg(["mean", "std"]).reset_index()
        agg.columns = [
            "_".join([str(v) for v in col if str(v) != ""]).rstrip("_")
            if isinstance(col, tuple) else col
            for col in agg.columns
        ]
        agg.to_csv(out_root / "aggregate_pruning.csv", index=False)

    p = out_root / "criterion_correlations.csv"
    if p.exists():
        df = pd.read_csv(p)
        agg = df.groupby(["target_class", "criterion"])["spearman_vs_ablation"].agg(
            ["mean", "std"]
        ).reset_index()
        agg.to_csv(out_root / "aggregate_correlations.csv", index=False)

    p = out_root / "path_similarity.csv"
    if p.exists():
        df = pd.read_csv(p)
        agg = df.groupby(["score_type"])[
            ["same_mean", "different_mean", "gap"]
        ].agg(["mean", "std"]).reset_index()
        agg.columns = [
            "_".join([str(v) for v in col if str(v) != ""]).rstrip("_")
            if isinstance(col, tuple) else col
            for col in agg.columns
        ]
        agg.to_csv(out_root / "aggregate_similarity.csv", index=False)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Do not run this experiment on CPU.")
    if torch.cuda.device_count() != 1:
        print(
            f"WARNING: {torch.cuda.device_count()} CUDA devices are visible. "
            "Expected one after CUDA_VISIBLE_DEVICES=1."
        )

    device = torch.device("cuda:0")  # physical GPU 1 after masking
    torch.set_float32_matmul_precision("high" if args.fast_matmul else "highest")

    if not (args.target_class == -1 or 0 <= args.target_class <= 9):
        raise ValueError("--target-class must be -1 or 0..9")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    env = save_environment(out_root)
    config = vars(args).copy()
    config["environment"] = env
    (out_root / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2))

    data = load_data(args.data_dir, device, args.cal_size, args.split_seed)
    print(
        "GPU-resident data:",
        {k: tuple(v.shape) for k, v in data.items()}
    )

    baseline_rows = []
    corr_rows = []
    sim_rows = []
    pruning_rows = []

    for seed in args.seeds:
        print(f"\n===== SEED {seed} =====")
        set_seed(seed)
        sd = out_root / f"seed_{seed}"
        sd.mkdir(parents=True, exist_ok=True)

        model = MLP().to(device)
        t0 = time.perf_counter()
        history = train(model, data, args, device)
        torch.cuda.synchronize()
        train_sec = time.perf_counter() - t0

        test = evaluate(
            model, data["x_test"], data["y_test"], args.eval_batch_size
        )
        cal = evaluate(
            model, data["x_cal"], data["y_cal"], args.eval_batch_size
        )
        print(
            f"baseline test={test['accuracy']:.4f}, "
            f"cal={cal['accuracy']:.4f}, train={train_sec:.2f}s"
        )

        torch.save(model.state_dict(), sd / "baseline.pt")
        save_csv(sd / "training_history.csv", history)

        base_row = {
            "seed": seed,
            "test_accuracy": test["accuracy"],
            "test_loss": test["loss"],
            "cal_accuracy": cal["accuracy"],
            "train_seconds": train_sec,
            "params": params(model),
            "macs_per_sample": macs(128),
        }
        for c, acc in enumerate(test["per_class_accuracy"]):
            base_row[f"class{c}_accuracy"] = acc
        baseline_rows.append(base_row)

        h2, logits = collect_h2(
            model, data["x_cal"], args.eval_batch_size
        )
        # Small enough (~5 MB) and valuable for later exploratory analysis.
        torch.save(
            {
                "h2": h2.detach().cpu(),
                "labels": data["y_cal"].detach().cpu(),
                "logits": logits.detach().cpu(),
            },
            sd / "calibration_h2.pt",
        )

        scores = build_scores(
            model, h2, data["y_cal"], args.target_class
        )
        global_ablation, class_ablation = exact_single_node_ablation(
            model, h2, data["y_cal"]
        )
        ablation_score = (
            global_ablation
            if args.target_class < 0
            else class_ablation[args.target_class]
        )

        # Node score table.
        node_rows = []
        for j in range(h2.size(1)):
            r = {
                "node": j,
                "weight": float(scores["weight"][j]),
                "activation": float(scores["activation"][j]),
                "weight_x_activation": float(scores["weight_x_activation"][j]),
                "grad_x_activation": float(scores["grad_x_activation"][j]),
                "ablation_delta_loss": float(ablation_score[j]),
                "incoming_norm": float(scores["incoming_norm"][j]),
                "outgoing_norm": float(scores["outgoing_norm"][j]),
            }
            for c in range(10):
                r[f"class{c}_mean_abs_activation"] = float(
                    scores["class_activation"][c, j]
                )
                r[f"class{c}_ablation_delta_loss"] = float(
                    class_ablation[c, j]
                )
            node_rows.append(r)
        save_csv(sd / "node_scores.csv", node_rows)

        np.savetxt(
            sd / "class_mean_abs_activation.csv",
            scores["class_activation"].cpu().numpy(),
            delimiter=",",
        )
        plot_heatmap(
            sd / "class_activation_heatmap.png",
            scores["class_activation"],
            "Mean absolute activation by class",
        )

        # Correlation against exact single-node ablation.
        for method in ["weight", "activation", "weight_x_activation", "grad_x_activation"]:
            rho = spearman(scores[method], ablation_score)
            corr_rows.append({
                "seed": seed,
                "target_class": args.target_class,
                "criterion": method,
                "spearman_vs_ablation": rho,
            })
            print(f"rho({method:19s}, ablation) = {rho:.4f}")

        # PRIMARY pathway similarity: label-independent score construction.
        similarity_scores = {
            "activation": h2.abs(),
            "activation_x_outnorm": (
                h2.abs() * scores["outgoing_norm"].view(1, -1)
            ),
        }
        for score_type, sample_scores in similarity_scores.items():
            mask, k = topk_masks(sample_scores, args.topk_fraction)
            js = pair_jaccard(
                mask, data["y_cal"], args.pair_samples, seed + 9001
            )
            sim_rows.append({
                "seed": seed,
                "score_type": score_type,
                "topk": k,
                **js,
            })
            print(
                f"Jaccard[{score_type}] same={js['same_mean']:.4f}, "
                f"diff={js['different_mean']:.4f}, gap={js['gap']:.4f}"
            )

            freq = class_topk_frequency(mask, data["y_cal"])
            np.savetxt(
                sd / f"class_topk_frequency_{score_type}.csv",
                freq.cpu().numpy(),
                delimiter=",",
            )
            plot_heatmap(
                sd / f"class_topk_frequency_{score_type}.png",
                freq,
                f"Top-k node frequency by class: {score_type}",
            )

        # Physical structured pruning.
        seed_pruning_rows = []
        for method in args.methods:
            for ratio in args.prune_ratios:
                keep = choose_keep(
                    method,
                    ratio,
                    scores,
                    ablation_score,
                    seed=seed*10000 + round(ratio*1000) + 17,
                    device=device,
                )
                pruned = PrunedMLP(model, keep).to(device)
                metrics = evaluate(
                    pruned,
                    data["x_test"],
                    data["y_test"],
                    args.eval_batch_size,
                )
                k = keep.numel()

                row = {
                    "seed": seed,
                    "target_class": args.target_class,
                    "method": method,
                    "prune_ratio": ratio,
                    "kept_nodes": k,
                    "removed_nodes": 128-k,
                    "test_accuracy": metrics["accuracy"],
                    "test_loss": metrics["loss"],
                    "params": params(pruned),
                    "param_reduction": 1 - params(pruned)/params(model),
                    "macs_per_sample": macs(k),
                    "mac_reduction": 1 - macs(k)/macs(128),
                }
                for c, acc in enumerate(metrics["per_class_accuracy"]):
                    row[f"class{c}_accuracy"] = acc

                if args.latency:
                    for b in args.latency_batches:
                        row[f"latency_ms_bs{b}"] = latency_ms(
                            pruned, b, args.latency_warmup,
                            args.latency_iters, device
                        )

                pruning_rows.append(row)
                seed_pruning_rows.append(row)
                print(
                    f"{method:20s} prune={ratio:.2f} h2={k:3d} "
                    f"acc={metrics['accuracy']:.4f} "
                    f"MAC↓={100*row['mac_reduction']:.1f}%"
                )

        save_csv(sd / "pruning_results.csv", seed_pruning_rows)
        plot_pruning(sd / "pruning_accuracy.png", seed_pruning_rows)

        del model, h2, logits, global_ablation, class_ablation
        torch.cuda.empty_cache()

    save_csv(out_root / "baseline_summary.csv", baseline_rows)
    save_csv(out_root / "criterion_correlations.csv", corr_rows)
    save_csv(out_root / "path_similarity.csv", sim_rows)
    save_csv(out_root / "all_pruning_results.csv", pruning_rows)
    aggregate_tables(out_root)

    print("\n===== COMPLETE =====")
    print("Results:", out_root.resolve())


if __name__ == "__main__":
    main()

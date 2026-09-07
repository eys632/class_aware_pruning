#!/usr/bin/env python3
"""
analyze_results.py
===================

`experiment.py` (mnist_structured_pruning_v2) 실행 결과 폴더(--out-dir로 지정한 곳,
예: runs/global_5seeds)를 입력으로 받아 Q1~Q3 세 가지 연구 질문에 대응하는
그림들을 한 번에 생성합니다.

사용법
------
    python analyze_results.py --run-dir runs/global_5seeds --out-dir figs/global_5seeds

여러 run(global vs class2 등)을 비교하고 싶으면 --compare-dir를 추가로 지정하세요.
    python analyze_results.py --run-dir runs/global_5seeds \
        --compare-dir runs/class2_5seeds --compare-label "class2" \
        --out-dir figs/compare

필요 패키지: pandas, matplotlib, seaborn (없으면 matplotlib만으로도 동작)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAVE_SNS = True
except ImportError:
    HAVE_SNS = False


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def load_csv(run_dir: Path, name: str) -> pd.DataFrame | None:
    p = run_dir / name
    if not p.exists():
        print(f"[skip] {p} 없음")
        return None
    return pd.read_csv(p)


def savefig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Q3. 실제 pruning으로 accuracy가 유지되는가
#     -> accuracy vs prune ratio (method별, seed 분산 포함)
#     -> accuracy vs MAC reduction (같은 압축률에서 실질 효율 비교)
#     -> class별 accuracy 저하 heatmap (특정 class만 붕괴하는지)
# ---------------------------------------------------------------------------

def plot_pruning_accuracy_vs_ratio(all_pruning: pd.DataFrame, out_dir: Path):
    """method별 mean accuracy 곡선 + seed 간 표준편차를 음영으로 표시.
    README의 pruning_accuracy.png와 같은 내용이지만, 여러 seed를 평균+분산으로
    합쳐서 '한 seed의 우연'이 아님을 확인하는 용도."""
    methods = sorted(all_pruning["method"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in methods:
        sub = all_pruning[all_pruning["method"] == m]
        g = sub.groupby("prune_ratio")["test_accuracy"].agg(["mean", "std"]).reset_index()
        x = g["prune_ratio"] * 100
        y = g["mean"] * 100
        yerr = g["std"].fillna(0) * 100
        ax.plot(x, y, marker="o", label=m)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)
    ax.set_xlabel("Pruned h2 nodes (%)")
    ax.set_ylabel("Test accuracy (%), mean +/- SD across seeds")
    ax.set_title("Accuracy Retention by Pruning Method (Mean Across Seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, out_dir, "q3_accuracy_vs_prune_ratio.png")


def plot_accuracy_vs_mac_reduction(all_pruning: pd.DataFrame, out_dir: Path):
    """같은 '실제 연산량 절감(MAC reduction)' 기준으로 method를 비교.
    prune_ratio는 h2 노드 비율일 뿐이고 실제 목표는 MAC/파라미터 절감이므로,
    x축을 mac_reduction으로 바꿔보면 어떤 방법이 '같은 효율 절감 대비 정확도'가
    더 좋은지 더 정확히 보인다."""
    methods = sorted(all_pruning["method"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in methods:
        sub = all_pruning[all_pruning["method"] == m]
        g = sub.groupby("prune_ratio").agg(
            mac=("mac_reduction", "mean"),
            acc=("test_accuracy", "mean"),
        ).reset_index().sort_values("mac")
        ax.plot(g["mac"] * 100, g["acc"] * 100, marker="o", label=m)
    ax.set_xlabel("MAC reduction (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. MAC Reduction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, out_dir, "q3_accuracy_vs_mac_reduction.png")


def plot_class_accuracy_heatmap(all_pruning: pd.DataFrame, out_dir: Path, method: str = "activation"):
    """특정 method에서 prune_ratio가 커질 때 class별 accuracy가 균일하게
    떨어지는지, 아니면 특정 class(예: 자주 헷갈리는 4/9, 3/5 등)만 먼저
    무너지는지 확인. class 붕괴 패턴은 '어떤 노드가 어떤 class를 담당하는지'
    (Q1)와 바로 연결되는 진단적 그림이다."""
    class_cols = [c for c in all_pruning.columns if c.startswith("class") and c.endswith("_accuracy")]
    if not class_cols:
        return
    sub = all_pruning[all_pruning["method"] == method]
    g = sub.groupby("prune_ratio")[class_cols].mean()
    g.columns = [c.replace("class", "").replace("_accuracy", "") for c in g.columns]
    g = g[sorted(g.columns, key=int)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(g.values.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(g.columns)))
    ax.set_yticklabels(g.columns)
    ax.set_ylabel("Digit class")
    ax.set_xticks(range(len(g.index)))
    ax.set_xticklabels([f"{r*100:.0f}%" for r in g.index])
    ax.set_xlabel("Prune ratio")
    ax.set_title(f"Class-wise Accuracy (Method: {method}, Mean Across Seeds)")
    fig.colorbar(im, ax=ax, label="Accuracy")
    savefig(fig, out_dir, f"q3_class_accuracy_heatmap_{method}.png")


# ---------------------------------------------------------------------------
# Q2. 어떤 score가 실제 ablation과 가장 잘 맞는가
#     -> criterion별 Spearman rho 막대그래프 (seed 오차막대)
#     -> (선택) 한 seed의 node_scores.csv에서 score vs ablation 산점도
# ---------------------------------------------------------------------------

def plot_criterion_correlation_bar(corr: pd.DataFrame, out_dir: Path):
    g = corr.groupby("criterion")["spearman_vs_ablation"].agg(["mean", "std"]).reset_index()
    g = g.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(g["criterion"], g["mean"], yerr=g["std"].fillna(0), capsize=4)
    ax.set_ylabel("Spearman correlation with exact single-node ablation")
    ax.set_title("Importance Score Agreement with Ablation Impact")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(min(0, g["mean"].min() - 0.1), 1.0)
    plt.xticks(rotation=20, ha="right")
    savefig(fig, out_dir, "q2_criterion_correlation_bar.png")


def plot_node_score_scatter(node_scores: pd.DataFrame, out_dir: Path, seed_label: str):
    """한 seed의 node_scores.csv로 그리는 산점도.
    막대그래프(요약 수치)만으로는 '왜' 어떤 score가 이기는지 감이 안 오는데,
    산점도를 보면 outlier node(예: 특정 클래스 전담 노드)가 상관계수를
    끌어올리거나 끌어내리는지 눈으로 확인할 수 있다."""
    criteria = ["weight", "activation", "weight_x_activation", "grad_x_activation"]
    criteria = [c for c in criteria if c in node_scores.columns]
    n = len(criteria)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, criteria):
        ax.scatter(node_scores[c], node_scores["ablation_delta_loss"], s=14, alpha=0.7)
        ax.set_xlabel(c)
        ax.set_title(c, fontsize=9)
    axes[0].set_ylabel("Ablation delta loss (measured importance)")
    fig.suptitle(f"Node-Level Score vs. Ablation Impact ({seed_label})")
    savefig(fig, out_dir, f"q2_node_score_scatter_{seed_label}.png")


# ---------------------------------------------------------------------------
# Q1. 같은 class가 비슷한 hidden node 집합을 쓰는가
#     -> same-class vs different-class Jaccard 분포 비교
#     -> class x node top-k frequency heatmap (seed 평균)
# ---------------------------------------------------------------------------

def plot_similarity_same_vs_diff(sim: pd.DataFrame, out_dir: Path):
    score_types = sim["score_type"].unique()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    width = 0.35
    x = np.arange(len(score_types))
    same_means = [sim[sim.score_type == s]["same_mean"].mean() for s in score_types]
    same_stds = [sim[sim.score_type == s]["same_mean"].std() for s in score_types]
    diff_means = [sim[sim.score_type == s]["different_mean"].mean() for s in score_types]
    diff_stds = [sim[sim.score_type == s]["different_mean"].std() for s in score_types]

    ax.bar(x - width/2, same_means, width, yerr=same_stds, capsize=4, label="same-class pair")
    ax.bar(x + width/2, diff_means, width, yerr=diff_stds, capsize=4, label="different-class pair")
    ax.set_xticks(x)
    ax.set_xticklabels(score_types)
    ax.set_ylabel("Top-k node set Jaccard similarity")
    ax.set_title("Hidden-Node Similarity by Class Pair (Mean Across Seeds)")
    ax.legend()
    savefig(fig, out_dir, "q1_similarity_same_vs_diff.png")


def plot_similarity_gap_by_seed(sim: pd.DataFrame, out_dir: Path):
    """gap(=same_mean - different_mean)이 5개 seed에서 항상 양수이고
    비슷한 크기인지 확인. seed마다 들쭉날쭉하면 '우연히 한 번 나온 결과'일
    수 있으므로 반드시 봐야 하는 진단 그림."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for score_type, sub in sim.groupby("score_type"):
        ax.plot(sub["seed"], sub["gap"], marker="o", label=score_type)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("seed")
    ax.set_ylabel("gap = same_mean - different_mean")
    ax.set_title("Consistency of the Same-vs.-Different-Class Gap Across Seeds")
    ax.legend()
    savefig(fig, out_dir, "q1_similarity_gap_by_seed.png")


def plot_topk_frequency_seed_avg(run_dir: Path, out_dir: Path, score_type: str = "activation"):
    """여러 seed의 class_topk_frequency_{score_type}.csv를 평균 내서,
    노드 초기화에 따른 우연이 아니라 seed에 걸쳐 일관되게 특정 노드가
    특정 class에서 자주 top-k에 뽑히는지를 본다."""
    mats = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        f = seed_dir / f"class_topk_frequency_{score_type}.csv"
        if f.exists():
            mats.append(np.loadtxt(f, delimiter=","))
    if not mats:
        print(f"[skip] class_topk_frequency_{score_type}.csv 없음")
        return
    avg = np.mean(np.stack(mats), axis=0)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(avg, aspect="auto")
    ax.set_xlabel("Hidden node index")
    ax.set_ylabel("Class")
    ax.set_yticks(range(avg.shape[0]))
    ax.set_title(
        f"Top-k Node Selection Frequency by Class "
        f"(Mean Across Seeds, Score: {score_type})"
    )
    fig.colorbar(im, ax=ax)
    savefig(fig, out_dir, f"q1_topk_frequency_seed_avg_{score_type}.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(run_dir: Path, out_dir: Path):
    print(f"\n=== {run_dir} 분석 ===")

    all_pruning = load_csv(run_dir, "all_pruning_results.csv")
    if all_pruning is not None:
        plot_pruning_accuracy_vs_ratio(all_pruning, out_dir)
        plot_accuracy_vs_mac_reduction(all_pruning, out_dir)
        for m in ["activation", "grad_x_activation", "random"]:
            if m in all_pruning["method"].unique():
                plot_class_accuracy_heatmap(all_pruning, out_dir, method=m)

    corr = load_csv(run_dir, "criterion_correlations.csv")
    if corr is not None:
        plot_criterion_correlation_bar(corr, out_dir)

    # 첫 seed 하나만 산점도로 (모든 seed를 다 그리면 과밀해짐)
    seed_dirs = sorted(run_dir.glob("seed_*"))
    if seed_dirs:
        ns = load_csv(seed_dirs[0], "node_scores.csv")
        if ns is not None:
            plot_node_score_scatter(ns, out_dir, seed_label=seed_dirs[0].name)

    sim = load_csv(run_dir, "path_similarity.csv")
    if sim is not None:
        plot_similarity_same_vs_diff(sim, out_dir)
        plot_similarity_gap_by_seed(sim, out_dir)

    for score_type in ["activation", "activation_x_outnorm"]:
        plot_topk_frequency_seed_avg(run_dir, out_dir, score_type=score_type)

    print(f"완료. 그림은 {out_dir} 에 저장됨.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                     help="experiment.py의 --out-dir (예: runs/global_5seeds)")
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="그림을 저장할 폴더")
    args = ap.parse_args()
    run(args.run_dir, args.out_dir)


if __name__ == "__main__":
    main()

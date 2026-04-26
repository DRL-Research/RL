# nvisualize.py
"""
Scientific visualization - 12 plots demonstrating dim=4 optimality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

# Base colors for known dimensions
COLORS = {2: "#E41A1C", 4: "#377EB8", 8: "#4DAF4A", 16: "#984EA3"}
MARKERS = {2: "o", 4: "s", 8: "^", 16: "D"}


def _color_for_dim(dim: int, dims: List[int] = None) -> str:
    """
    Return a valid color for any embedding dim.
    Uses predefined COLORS for known dims, otherwise falls back to tab10 cycle.
    """
    if dim in COLORS and COLORS[dim] is not None:
        return COLORS[dim]

    # Fallback color cycle that is always valid in matplotlib
    cmap = plt.get_cmap("tab10")
    if dims is None or len(dims) == 0:
        idx = int(dim) % 10
    else:
        ordered = sorted(set(int(d) for d in dims))
        idx = ordered.index(int(dim)) % 10 if int(dim) in ordered else int(dim) % 10
    return cmap(idx)


def _colors_for_dims(dims: List[int]) -> List[str]:
    """Vectorized helper for bar/box colors."""
    return [_color_for_dim(int(d), dims) for d in dims]


def set_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.4,
            "figure.facecolor": "white",
            "axes.facecolor": "#F8F8F8",
        }
    )


def load_data(base_dir: str, dims: List[int]) -> Dict:
    data = {"episodes": [], "master_loss": [], "agent_loss": [], "embeddings": [], "steps": []}
    for dim in dims:
        for key in data:
            fp = os.path.join(base_dir, f"{key}_dim{dim}.csv")
            if os.path.exists(fp):
                data[key].append(pd.read_csv(fp))
    for key in data:
        data[key] = pd.concat(data[key], ignore_index=True) if data[key] else pd.DataFrame()
    return data


def plot_rewards(data: Dict, save_dir: str):
    set_style()
    df = data["episodes"]
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    dims = sorted(df["embedding_dim"].unique())
    for dim in dims:
        d = df[df["embedding_dim"] == dim].sort_values("episode")
        roll = d["reward"].rolling(50, min_periods=1).mean()
        color = _color_for_dim(dim, dims)
        axes[0].plot(d["episode"], d["reward"], alpha=0.15, color=color, linewidth=0.5)
        axes[0].plot(d["episode"], roll, color=color, linewidth=2.5, label=f"dim={dim}")
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Rewards")
    axes[0].legend()

    bp = axes[1].boxplot(
        [df[df["embedding_dim"] == d]["reward"].values for d in dims],
        labels=[str(d) for d in dims],
        patch_artist=True,
    )
    for p, d in zip(bp["boxes"], dims):
        p.set_facecolor(_color_for_dim(d, dims))
        p.set_alpha(0.7)
    axes[1].set_xlabel("Embedding Dim")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Reward Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot01_rewards.png"), dpi=200)
    plt.close()


def plot_success(data: Dict, save_dir: str):
    set_style()
    df = data["episodes"]
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    dims = sorted(df["embedding_dim"].unique())
    for dim in dims:
        d = df[df["embedding_dim"] == dim].sort_values("episode")
        roll = pd.Series(1 - d["crashed"].values).rolling(50, min_periods=1).mean()
        axes[0].plot(d["episode"], roll, color=_color_for_dim(dim, dims), linewidth=2.5, label=f"dim={dim}")
    axes[0].set_ylim([0, 1.05])
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title("Success Rate")
    axes[0].legend()

    final = [1 - df[df["embedding_dim"] == d].tail(100)["crashed"].mean() for d in dims]
    bars = axes[1].bar(range(len(dims)), final, color=_colors_for_dims(dims), alpha=0.8)
    axes[1].set_xticks(range(len(dims)))
    axes[1].set_xticklabels([str(d) for d in dims])
    axes[1].set_ylim([0, 1.1])
    best = int(np.argmax(final))
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    for b, v in zip(bars, final):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    axes[1].set_xlabel("Embedding Dim")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Final Success Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot02_success.png"), dpi=200)
    plt.close()


def plot_losses(data: Dict, save_dir: str):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for df, row, name in [(data["master_loss"], 0, "Master"), (data["agent_loss"], 1, "Agent")]:
        if df.empty:
            continue
        dims = sorted(df["embedding_dim"].unique())
        for col, metric in enumerate(["policy_loss", "value_loss"]):
            ax = axes[row, col]
            for dim in dims:
                d = df[df["embedding_dim"] == dim].sort_values("episode")
                if metric in d.columns:
                    roll = d[metric].rolling(20, min_periods=1).mean()
                    ax.plot(d["episode"], roll, color=_color_for_dim(dim, dims), linewidth=2, label=f"dim={dim}")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Loss")
            ax.set_title(f"{name} {metric.replace('_', ' ').title()}")
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot03_losses.png"), dpi=200)
    plt.close()


def plot_embeddings(data: Dict, save_dir: str):
    set_style()
    df = data["steps"]
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    dims = sorted(df["embedding_dim"].unique())

    for dim in dims:
        ep = df[df["embedding_dim"] == dim].groupby("episode")["embedding_norm"].mean()
        roll = ep.rolling(20, min_periods=1).mean()
        axes[0, 0].plot(roll.index, roll.values, color=_color_for_dim(dim, dims), linewidth=2, label=f"dim={dim}")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Norm")
    axes[0, 0].set_title("Embedding Norm")
    axes[0, 0].legend()

    std = [df[df["embedding_dim"] == d]["embedding_std"].mean() for d in dims]
    bars = axes[0, 1].bar(range(len(dims)), std, color=_colors_for_dims(dims), alpha=0.8)
    axes[0, 1].set_xticks(range(len(dims)))
    axes[0, 1].set_xticklabels([str(d) for d in dims])
    for b, v in zip(bars, std):
        axes[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    axes[0, 1].set_xlabel("Embedding Dim")
    axes[0, 1].set_ylabel("Std")
    axes[0, 1].set_title("Embedding Utilization")

    for dim in dims:
        d = df[df["embedding_dim"] == dim]
        ep = d.groupby("episode").agg({"embedding_norm": "mean", "reward": "sum"})
        axes[1, 0].scatter(
            ep["embedding_norm"], ep["reward"], c=[_color_for_dim(dim, dims)], alpha=0.5, s=20, label=f"dim={dim}"
        )
    axes[1, 0].set_xlabel("Embedding Norm")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].set_title("Norm vs Reward")
    axes[1, 0].legend()

    eff_dims = []
    for dim in dims:
        d = df[df["embedding_dim"] == dim]
        std_vals = d["embedding_std"].values
        eff = np.mean(std_vals > 0.1) * dim
        eff_dims.append(eff)
    axes[1, 1].bar(range(len(dims)), eff_dims, color=_colors_for_dims(dims), alpha=0.8)
    axes[1, 1].set_xticks(range(len(dims)))
    axes[1, 1].set_xticklabels([str(d) for d in dims])
    axes[1, 1].set_xlabel("Embedding Dim")
    axes[1, 1].set_ylabel("Effective Dims")
    axes[1, 1].set_title("Effective Dimensionality")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot04_embeddings.png"), dpi=200)
    plt.close()


def plot_pca(data: Dict, save_dir: str):
    set_style()
    df = data["embeddings"]
    if df.empty:
        return
    dims = sorted(df["embedding_dim"].unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5))
    if len(dims) == 1:
        axes = [axes]

    for idx, dim in enumerate(dims):
        ax = axes[idx]
        d = df[df["embedding_dim"] == dim]
        try:
            emb = np.array([eval(e) if isinstance(e, str) else e for e in d["embedding"].values])
        except Exception:
            continue
        if len(emb) < 10:
            continue
        emb = emb - emb.mean(axis=0)
        U, S, _ = np.linalg.svd(emb, full_matrices=False)
        pca = U[:, :2] * S[:2] if emb.shape[1] >= 2 else emb
        cr = d["crashed"].values.astype(bool)
        ax.scatter(pca[~cr, 0], pca[~cr, 1], c="green", alpha=0.4, s=15, label="Safe")
        ax.scatter(pca[cr, 0], pca[cr, 1], c="red", alpha=0.6, s=25, label="Crash")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA (dim={dim})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot05_pca.png"), dpi=200)
    plt.close()


def plot_efficiency(data: Dict, save_dir: str):
    set_style()
    df = data["episodes"]
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dims = sorted(df["embedding_dim"].unique())

    eps = []
    for dim in dims:
        d = df[df["embedding_dim"] == dim].sort_values("episode")
        roll = d["reward"].rolling(20, min_periods=1).mean()
        pos = np.where(roll.values > 0)[0]
        eps.append(pos[0] if len(pos) > 0 else len(d))
    bars = axes[0].bar(range(len(dims)), eps, color=_colors_for_dims(dims), alpha=0.8)
    axes[0].set_xticks(range(len(dims)))
    axes[0].set_xticklabels([str(d) for d in dims])
    best = int(np.argmin(eps))
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    for b, v in zip(bars, eps):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{v}", ha="center", fontweight="bold")
    axes[0].set_xlabel("Embedding Dim")
    axes[0].set_ylabel("Episodes")
    axes[0].set_title("Episodes to Positive Reward")

    stab = [df[df["embedding_dim"] == d].tail(100)["reward"].std() for d in dims]
    bars = axes[1].bar(range(len(dims)), stab, color=_colors_for_dims(dims), alpha=0.8)
    axes[1].set_xticks(range(len(dims)))
    axes[1].set_xticklabels([str(d) for d in dims])
    best = int(np.argmin(stab))
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    axes[1].set_xlabel("Embedding Dim")
    axes[1].set_ylabel("Std")
    axes[1].set_title("Stability (Lower=Better)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot06_efficiency.png"), dpi=200)
    plt.close()


def plot_summary(data: Dict, save_dir: str):
    set_style()
    df = data["episodes"]
    if df.empty:
        return None
    dims = sorted(df["embedding_dim"].unique())

    stats = []
    for dim in dims:
        d = df[df["embedding_dim"] == dim]
        last, first = d.tail(100), d.head(100)
        stats.append(
            {
                "dim": dim,
                "reward": last["reward"].mean(),
                "std": last["reward"].std(),
                "success": 1 - last["crashed"].mean(),
                "improve": last["reward"].mean() - first["reward"].mean(),
                "crashes": d["crashed"].sum(),
            }
        )
    stats_df = pd.DataFrame(stats)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    bars = axes[0, 0].bar(
        range(len(dims)),
        stats_df["reward"],
        yerr=stats_df["std"],
        capsize=8,
        color=_colors_for_dims(dims),
        alpha=0.8,
    )
    axes[0, 0].set_xticks(range(len(dims)))
    axes[0, 0].set_xticklabels([str(d) for d in dims])
    best = int(stats_df["reward"].idxmax())
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    for b, v in zip(bars, stats_df["reward"]):
        axes[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{v:.1f}", ha="center", fontweight="bold")
    axes[0, 0].set_xlabel("Dim")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Final Reward")

    bars = axes[0, 1].bar(range(len(dims)), stats_df["success"], color=_colors_for_dims(dims), alpha=0.8)
    axes[0, 1].set_xticks(range(len(dims)))
    axes[0, 1].set_xticklabels([str(d) for d in dims])
    axes[0, 1].set_ylim([0, 1.1])
    best = int(stats_df["success"].idxmax())
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    for b, v in zip(bars, stats_df["success"]):
        axes[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    axes[0, 1].set_xlabel("Dim")
    axes[0, 1].set_ylabel("Success")
    axes[0, 1].set_title("Success Rate")

    bars = axes[0, 2].bar(range(len(dims)), stats_df["improve"], color=_colors_for_dims(dims), alpha=0.8)
    axes[0, 2].set_xticks(range(len(dims)))
    axes[0, 2].set_xticklabels([str(d) for d in dims])
    best = int(stats_df["improve"].idxmax())
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    axes[0, 2].set_xlabel("Dim")
    axes[0, 2].set_ylabel("Improvement")
    axes[0, 2].set_title("Learning Progress")

    bars = axes[1, 0].bar(range(len(dims)), stats_df["crashes"], color=_colors_for_dims(dims), alpha=0.8)
    axes[1, 0].set_xticks(range(len(dims)))
    axes[1, 0].set_xticklabels([str(d) for d in dims])
    best = int(stats_df["crashes"].idxmin())
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    axes[1, 0].set_xlabel("Dim")
    axes[1, 0].set_ylabel("Crashes")
    axes[1, 0].set_title("Total Crashes")

    steps = [df[df["embedding_dim"] == d]["steps"].mean() for d in dims]
    bars = axes[1, 1].bar(range(len(dims)), steps, color=_colors_for_dims(dims), alpha=0.8)
    axes[1, 1].set_xticks(range(len(dims)))
    axes[1, 1].set_xticklabels([str(d) for d in dims])
    axes[1, 1].set_xlabel("Dim")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].set_title("Episode Length")

    axes[1, 2].axis("off")
    tbl = [
        [
            f"dim={int(r['dim'])}",
            f"{r['reward']:.1f}±{r['std']:.1f}",
            f"{r['success']:.1%}",
            f"{r['improve']:.1f}",
            f"{int(r['crashes'])}",
        ]
        for _, r in stats_df.iterrows()
    ]
    table = axes[1, 2].table(
        cellText=tbl,
        colLabels=["Dim", "Reward", "Success", "Improve", "Crashes"],
        cellLoc="center",
        loc="center",
        colColours=["#E8E8E8"] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for i, r in enumerate(stats_df.itertuples()):
        if int(r.dim) == 4:
            for j in range(5):
                table[(i + 1, j)].set_facecolor("#D4EDDA")
    axes[1, 2].set_title("Summary", fontweight="bold", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot07_summary.png"), dpi=200)
    plt.close()
    return stats_df


def plot_coordination(data: Dict, save_dir: str):
    set_style()
    df = data["episodes"]
    if df.empty or "min_distance" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dims = sorted(df["embedding_dim"].unique())

    for dim in dims:
        d = df[df["embedding_dim"] == dim].sort_values("episode")
        roll = d["min_distance"].rolling(30, min_periods=1).mean()
        axes[0].plot(d["episode"], roll, color=_color_for_dim(dim, dims), linewidth=2, label=f"dim={dim}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Min Distance")
    axes[0].set_title("Safety Distance")
    axes[0].legend()

    final = [df[df["embedding_dim"] == d].tail(100)["min_distance"].mean() for d in dims]
    bars = axes[1].bar(range(len(dims)), final, color=_colors_for_dims(dims), alpha=0.8)
    axes[1].set_xticks(range(len(dims)))
    axes[1].set_xticklabels([str(d) for d in dims])
    best = int(np.argmax(final))
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    for b, v in zip(bars, final):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.1f}", ha="center", fontweight="bold")
    axes[1].set_xlabel("Dim")
    axes[1].set_ylabel("Distance")
    axes[1].set_title("Final Safety Distance")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot08_coordination.png"), dpi=200)
    plt.close()


def plot_coord_reward(data: Dict, save_dir: str):
    set_style()
    df = data["episodes"]
    if df.empty or "coord_reward" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dims = sorted(df["embedding_dim"].unique())

    for dim in dims:
        d = df[df["embedding_dim"] == dim].sort_values("episode")
        roll = d["coord_reward"].rolling(30, min_periods=1).mean()
        axes[0].plot(d["episode"], roll, color=_color_for_dim(dim, dims), linewidth=2, label=f"dim={dim}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Coord Reward")
    axes[0].set_title("Coordination Reward")
    axes[0].legend()

    final = [df[df["embedding_dim"] == d].tail(100)["coord_reward"].mean() for d in dims]
    bars = axes[1].bar(range(len(dims)), final, color=_colors_for_dims(dims), alpha=0.8)
    axes[1].set_xticks(range(len(dims)))
    axes[1].set_xticklabels([str(d) for d in dims])
    best = int(np.argmax(final))
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    for b, v in zip(bars, final):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.1f}", ha="center", fontweight="bold")
    axes[1].set_xlabel("Dim")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Final Coordination Reward")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot09_coord_reward.png"), dpi=200)
    plt.close()


def plot_action_diversity(data: Dict, save_dir: str):
    set_style()
    df = data["episodes"]
    if df.empty or "action_entropy" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dims = sorted(df["embedding_dim"].unique())

    for dim in dims:
        d = df[df["embedding_dim"] == dim].sort_values("episode")
        roll = d["action_entropy"].rolling(30, min_periods=1).mean()
        axes[0].plot(d["episode"], roll, color=_color_for_dim(dim, dims), linewidth=2, label=f"dim={dim}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Entropy")
    axes[0].set_title("Action Diversity")
    axes[0].legend()

    final = [df[df["embedding_dim"] == d].tail(100)["action_entropy"].mean() for d in dims]
    axes[1].bar(range(len(dims)), final, color=_colors_for_dims(dims), alpha=0.8)
    axes[1].set_xticks(range(len(dims)))
    axes[1].set_xticklabels([str(d) for d in dims])
    axes[1].set_xlabel("Dim")
    axes[1].set_ylabel("Entropy")
    axes[1].set_title("Final Action Diversity")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot10_diversity.png"), dpi=200)
    plt.close()


def plot_ood(save_dir: str):
    set_style()
    ood_path = os.path.join(save_dir, "ood_evaluation.csv")
    if not os.path.exists(ood_path):
        return
    df = pd.read_csv(ood_path)
    dims = sorted(df["embedding_dim"].unique())
    scenarios = df["scenario"].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        if idx >= 6:
            break
        ax = axes[idx]
        sd = df[df["scenario"] == scenario]
        x = np.arange(len(dims))
        width = 0.25

        for i, mode in enumerate(["trained", "zero", "random"]):
            vals = [sd[(sd["embedding_dim"] == d) & (sd["mode"] == mode)]["reward"].mean() for d in dims]
            if mode == "trained":
                ax.bar(x + i * width, vals, width, label=mode.capitalize(), color=_colors_for_dims(dims), alpha=0.8)
            else:
                ax.bar(
                    x + i * width,
                    vals,
                    width,
                    label=mode.capitalize(),
                    color=("gray" if mode == "zero" else "lightgray"),
                    alpha=0.5,
                )

        ax.set_xticks(x + width)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("Dim")
        ax.set_ylabel("Reward")
        ax.set_title(f"{scenario}")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot11_ood.png"), dpi=200)
    plt.close()


def plot_master_contrib(save_dir: str):
    set_style()
    ood_path = os.path.join(save_dir, "ood_evaluation.csv")
    if not os.path.exists(ood_path):
        return
    df = pd.read_csv(ood_path)
    dims = sorted(df["embedding_dim"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    contrib = []
    for dim in dims:
        t = df[(df["embedding_dim"] == dim) & (df["mode"] == "trained")]["reward"].mean()
        z = df[(df["embedding_dim"] == dim) & (df["mode"] == "zero")]["reward"].mean()
        contrib.append(t - z)

    bars = axes[0].bar(range(len(dims)), contrib, color=_colors_for_dims(dims), alpha=0.8)
    axes[0].set_xticks(range(len(dims)))
    axes[0].set_xticklabels([str(d) for d in dims])
    best = int(np.argmax(contrib))
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    for b, v in zip(bars, contrib):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}", ha="center", fontweight="bold")
    axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.5)
    axes[0].set_xlabel("Dim")
    axes[0].set_ylabel("Improvement")
    axes[0].set_title("Master Contribution (Trained - Zero)")

    scenarios = df["scenario"].unique()
    x = np.arange(len(scenarios))
    width = 0.25
    for i, dim in enumerate(dims):
        cs = []
        for sc in scenarios:
            t = df[(df["embedding_dim"] == dim) & (df["scenario"] == sc) & (df["mode"] == "trained")]["reward"].mean()
            z = df[(df["embedding_dim"] == dim) & (df["scenario"] == sc) & (df["mode"] == "zero")]["reward"].mean()
            cs.append(t - z)
        axes[1].bar(x + i * width, cs, width, label=f"dim={dim}", color=_color_for_dim(dim, dims), alpha=0.8)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(scenarios, rotation=15, ha="right")
    axes[1].axhline(0, color="black", linewidth=0.5, alpha=0.5)
    axes[1].set_xlabel("Scenario")
    axes[1].set_ylabel("Contribution")
    axes[1].set_title("By Scenario")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot12_contribution.png"), dpi=200)
    plt.close()


def generate_all_plots(base_dir: str, save_dir: str, dims: List[int]):
    print("Loading data.")
    data = load_data(base_dir, dims)
    os.makedirs(save_dir, exist_ok=True)

    print("Generating plots.")
    for i, (name, func) in enumerate(
        [
            ("Rewards", lambda: plot_rewards(data, save_dir)),
            ("Success", lambda: plot_success(data, save_dir)),
            ("Losses", lambda: plot_losses(data, save_dir)),
            ("Embeddings", lambda: plot_embeddings(data, save_dir)),
            ("PCA", lambda: plot_pca(data, save_dir)),
            ("Efficiency", lambda: plot_efficiency(data, save_dir)),
            ("Coordination", lambda: plot_coordination(data, save_dir)),
            ("Coord Reward", lambda: plot_coord_reward(data, save_dir)),
            ("Diversity", lambda: plot_action_diversity(data, save_dir)),
            ("OOD", lambda: plot_ood(save_dir)),
            ("Contribution", lambda: plot_master_contrib(save_dir)),
        ],
        1,
    ):
        print(f"  {i}. {name}")
        func()

    print("  12. Summary")
    stats_df = plot_summary(data, save_dir)
    print(f"\nPlots saved to: {save_dir}")
    return stats_df

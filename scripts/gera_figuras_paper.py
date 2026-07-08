from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURAÇÕES
# =========================================================

BASE_DIR = Path("saida_estudo_completo_PCA_V2")
OUTPUT_DIR = BASE_DIR / "figuras_paper"

DATASETS = ["600", "438", "442"]

CLASS_COLORS = {
    1: "#1f77b4",  # blue
    2: "#2ca02c",  # green
    3: "#d62728",  # red
}

DATASET_TITLES = {
    "600": "600-feature sequence embedding",
    "438": "438 features",
    "442": "442 features",
}


# =========================================================
# AUXILIARES
# =========================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_class_means_data(dataset_name: str):
    dataset_dir = BASE_DIR / f"saida_pca_classes_{dataset_name}"

    pca_path = dataset_dir / f"pca_of_class_means_{dataset_name}.csv"
    evr_path = dataset_dir / f"pca_explained_variance_{dataset_name}.csv"

    if not pca_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pca_path.resolve()}")
    if not evr_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {evr_path.resolve()}")

    pca_df = pd.read_csv(pca_path, encoding="utf-8")
    evr_df = pd.read_csv(evr_path, encoding="utf-8")

    return pca_df, evr_df


def load_sampled_data(dataset_name: str):
    dataset_dir = BASE_DIR / f"saida_pca_classes_{dataset_name}"

    pca_path = dataset_dir / f"pca_scatter_sampled_{dataset_name}.csv"
    evr_path = dataset_dir / f"pca_explained_variance_sampled_{dataset_name}.csv"

    if not pca_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pca_path.resolve()}")
    if not evr_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {evr_path.resolve()}")

    pca_df = pd.read_csv(pca_path, encoding="utf-8")
    evr_df = pd.read_csv(evr_path, encoding="utf-8")

    return pca_df, evr_df


def get_pc_variance(evr_df: pd.DataFrame) -> tuple[float, float]:
    evr_pc1 = float(
        evr_df.loc[evr_df["Component"] == "PC1", "ExplainedVariancePercent"].iloc[0]
    )
    evr_pc2 = float(
        evr_df.loc[evr_df["Component"] == "PC2", "ExplainedVariancePercent"].iloc[0]
    )
    return evr_pc1, evr_pc2


# =========================================================
# FIGURA 1 - CLASS CENTROIDS
# =========================================================

def build_figure_1_class_centroids() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    for ax, dataset_name in zip(axes, DATASETS):
        pca_df, evr_df = load_class_means_data(dataset_name)
        evr_pc1, evr_pc2 = get_pc_variance(evr_df)

        for _, row in pca_df.iterrows():
            cls = int(row["Class"])
            ax.scatter(
                row["PC1"],
                row["PC2"],
                s=220,
                color=CLASS_COLORS.get(cls, "#333333"),
            )
            ax.annotate(
                f"Class {cls}",
                (row["PC1"], row["PC2"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=10,
            )

        ax.set_title(DATASET_TITLES[dataset_name], fontsize=11)
        ax.set_xlabel(f"PC1 ({evr_pc1:.2f}%)")
        ax.set_ylabel(f"PC2 ({evr_pc2:.2f}%)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "PCA of class centroids after removing class 4",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    out_png = OUTPUT_DIR / "Fig1_PCA_class_centroids.png"
    out_pdf = OUTPUT_DIR / "Fig1_PCA_class_centroids.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure 1 saved to: {out_png.resolve()}")
    print(f"[OK] Figure 1 saved to: {out_pdf.resolve()}")


# =========================================================
# FIGURA 2 - SAMPLED PCA
# =========================================================

def build_figure_2_sampled_pca() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8))

    for ax, dataset_name in zip(axes, DATASETS):
        pca_df, evr_df = load_sampled_data(dataset_name)
        evr_pc1, evr_pc2 = get_pc_variance(evr_df)

        for cls in sorted(pca_df["Class"].unique()):
            cls = int(cls)
            cls_df = pca_df[pca_df["Class"] == cls]

            ax.scatter(
                cls_df["PC1"],
                cls_df["PC2"],
                s=4,
                alpha=0.28,
                color=CLASS_COLORS.get(cls, "#333333"),
                label=f"Class {cls}",
            )

        ax.set_title(DATASET_TITLES[dataset_name], fontsize=11)
        ax.set_xlabel(f"PC1 ({evr_pc1:.2f}%)")
        ax.set_ylabel(f"PC2 ({evr_pc2:.2f}%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Sampled PCA projections for the three datasets",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    out_png = OUTPUT_DIR / "Fig2_Sampled_PCA.png"
    out_pdf = OUTPUT_DIR / "Fig2_Sampled_PCA.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure 2 saved to: {out_png.resolve()}")
    print(f"[OK] Figure 2 saved to: {out_pdf.resolve()}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("[INFO] Building paper figures...")
    build_figure_1_class_centroids()
    build_figure_2_sampled_pca()
    print("[SUCCESS] Figures generated in figuras_paper.")


if __name__ == "__main__":
    main()
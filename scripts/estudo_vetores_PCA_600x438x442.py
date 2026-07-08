from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURAÇÕES
# =========================================================

DIR_600 = Path("saida_valida_features/saida_pca_classes_600")
DIR_438 = Path("saida_valida_features/saida_pca_classes_438")
DIR_442 = Path("saida_valida_features/saida_pca_classes_442")

OUTPUT_DIR = Path("saida_pca_comparacao_sem_classe4")
OUTPUT_IMG = OUTPUT_DIR / "comparacao_pca_classes_600_vs_438_vs_442_sem_classe4.png"


# =========================================================
# AUXILIARES
# =========================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pca_data(base_dir: Path, suffix: str):
    pca_path = base_dir / f"pca_of_class_means_{suffix}.csv"
    evr_path = base_dir / f"pca_explained_variance_{suffix}.csv"

    if not pca_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pca_path.resolve()}")

    if not evr_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {evr_path.resolve()}")

    pca_df = pd.read_csv(pca_path, encoding="utf-8")
    evr_df = pd.read_csv(evr_path, encoding="utf-8")

    return pca_df, evr_df


# =========================================================
# PLOT
# =========================================================

def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("[INFO] Carregando PCA das 600 features...")
    pca_600, evr_600 = load_pca_data(DIR_600, "600")

    print("[INFO] Carregando PCA das 438 features...")
    pca_438, evr_438 = load_pca_data(DIR_438, "438")

    print("[INFO] Carregando PCA das 442 features...")
    pca_442, evr_442 = load_pca_data(DIR_442, "442")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    datasets = [
        ("600 features", pca_600, evr_600),
        ("438 features", pca_438, evr_438),
        ("442 features", pca_442, evr_442),
    ]

    for ax, (title, pca_df, evr_df) in zip(axes, datasets):
        if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns:
            raise ValueError(f"O arquivo PCA de {title} não contém PC1 e PC2.")

        evr_pc1 = float(evr_df.loc[evr_df["Component"] == "PC1", "ExplainedVariancePercent"].iloc[0])
        evr_pc2 = float(evr_df.loc[evr_df["Component"] == "PC2", "ExplainedVariancePercent"].iloc[0])

        ax.scatter(pca_df["PC1"], pca_df["PC2"], s=180)

        for _, row in pca_df.iterrows():
            ax.annotate(
                f"Classe {int(row['Class'])}",
                (row["PC1"], row["PC2"]),
                textcoords="offset points",
                xytext=(8, 8),
            )

        ax.set_title(f"PCA dos vetores médios - {title}\n(classes 1, 2 e 3)")
        ax.set_xlabel(f"PC1 ({evr_pc1:.2f}%)")
        ax.set_ylabel(f"PC2 ({evr_pc2:.2f}%)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    plt.close()

    print("\n[SUCESSO] Imagem comparativa gerada.")
    print(f"[INFO] Arquivo salvo em: {OUTPUT_IMG.resolve()}")


if __name__ == "__main__":
    main()
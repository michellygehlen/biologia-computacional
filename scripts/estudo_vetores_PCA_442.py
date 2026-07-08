from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURAÇÕES
# =========================================================

#FEATURES_DIR = Path("saida_features_aa_dipep_v3_seed_6milhoes") <--- tá errado... não rodei com 6 milhões ainda prá 442
#OUTPUT_DIR = Path("saida_pca_classes_442") <- usar qdo rodar prá 6 milhões
FEATURES_DIR = Path("saida_features_aa_dipep_v4_442")
OUTPUT_DIR = Path("saida_pca_classes_442_amostra")

STANDARDIZE_BEFORE_PCA = True


# =========================================================
# AUXILIARES
# =========================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def load_memmaps(features_dir: Path):
    meta_path = features_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json não encontrado em: {features_dir.resolve()}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    x_path = Path(meta["x_memmap_path"])
    y_path = Path(meta["y_memmap_path"])
    shape_X = tuple(meta["shape_X"])
    shape_y = tuple(meta["shape_y"])

    X = np.memmap(
        x_path,
        dtype=np.float32,
        mode="r",
        shape=shape_X,
    )

    y = np.memmap(
        y_path,
        dtype=np.int32,
        mode="r",
        shape=shape_y,
    )

    return X, y, meta


def load_feature_names(features_dir: Path, n_features: int) -> list[str]:
    feature_names_path = features_dir / "feature_names.txt"

    if feature_names_path.exists():
        with feature_names_path.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == n_features:
            return names

    return [f"F{i:03d}" for i in range(n_features)]


# =========================================================
# PIPELINE
# =========================================================

def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print(f"[INFO] Carregando memmaps de: {FEATURES_DIR.resolve()}")
    X, y, meta = load_memmaps(FEATURES_DIR)

    n_rows, n_features = X.shape
    print(f"[INFO] Shape X: {X.shape}")
    print(f"[INFO] Shape y: {y.shape}")

    classes = sorted(np.unique(np.asarray(y)).tolist())
    print(f"[INFO] Classes encontradas: {classes}")

    feature_names = load_feature_names(FEATURES_DIR, n_features)

    mean_rows = []
    min_rows = []
    max_rows = []
    summary_rows = []

    y_array = np.asarray(y)

    for cls in classes:
        print(f"[INFO] Processando classe {cls} ...")
        idx = np.where(y_array == cls)[0]

        if idx.size == 0:
            continue

        X_cls = np.asarray(X[idx], dtype=np.float64)

        mean_vec = X_cls.mean(axis=0)
        min_vec = X_cls.min(axis=0)
        max_vec = X_cls.max(axis=0)

        mean_row = {"Class": cls, "N": int(idx.size)}
        min_row = {"Class": cls, "N": int(idx.size)}
        max_row = {"Class": cls, "N": int(idx.size)}

        for i, col in enumerate(feature_names):
            mean_row[col] = float(mean_vec[i])
            min_row[col] = float(min_vec[i])
            max_row[col] = float(max_vec[i])

        mean_rows.append(mean_row)
        min_rows.append(min_row)
        max_rows.append(max_row)

        summary_rows.append({
            "Class": cls,
            "N": int(idx.size),
        })

    mean_df = pd.DataFrame(mean_rows)
    min_df = pd.DataFrame(min_rows)
    max_df = pd.DataFrame(max_rows)
    summary_df = pd.DataFrame(summary_rows)

    save_dataframe(mean_df, OUTPUT_DIR / "class_mean_vectors_442.csv")
    save_dataframe(min_df, OUTPUT_DIR / "class_min_vectors_442.csv")
    save_dataframe(max_df, OUTPUT_DIR / "class_max_vectors_442.csv")
    save_dataframe(summary_df, OUTPUT_DIR / "class_counts_442.csv")

    print("[INFO] Vetores médios, mínimos e máximos salvos.")

    # =====================================================
    # PCA dos 4 vetores médios
    # =====================================================
    mean_matrix = mean_df[feature_names].to_numpy(dtype=np.float64)

    if STANDARDIZE_BEFORE_PCA:
        scaler = StandardScaler()
        mean_matrix_for_pca = scaler.fit_transform(mean_matrix)
    else:
        mean_matrix_for_pca = mean_matrix

    n_components = min(3, len(classes), n_features)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(mean_matrix_for_pca)

    pca_cols = [f"PC{i+1}" for i in range(coords.shape[1])]
    pca_df = pd.DataFrame(coords, columns=pca_cols)
    pca_df.insert(0, "Class", mean_df["Class"].values)
    pca_df.insert(1, "N", mean_df["N"].values)

    save_dataframe(pca_df, OUTPUT_DIR / "pca_of_class_means_442.csv")

    explained_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "ExplainedVarianceRatio": pca.explained_variance_ratio_,
        "ExplainedVariancePercent": pca.explained_variance_ratio_ * 100.0,
    })
    save_dataframe(explained_df, OUTPUT_DIR / "pca_explained_variance_442.csv")

    # =====================================================
    # Gráfico 2D
    # =====================================================
    if coords.shape[1] >= 2:
        plt.figure(figsize=(8, 6))

        x = coords[:, 0]
        y_plot = coords[:, 1]

        plt.scatter(x, y_plot, s=140)

        for i, cls in enumerate(mean_df["Class"].tolist()):
            plt.annotate(
                f"Classe {cls}",
                (x[i], y_plot[i]),
                textcoords="offset points",
                xytext=(8, 8),
            )

        evr = pca.explained_variance_ratio_
        plt.xlabel(f"PC1 ({evr[0] * 100:.2f}%)")
        plt.ylabel(f"PC2 ({evr[1] * 100:.2f}%)")
        plt.title("PCA dos vetores médios das classes (442 features)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pca_of_class_means_442_2d.png", dpi=200)
        plt.close()

        print("[INFO] Gráfico PCA 2D salvo.")
    else:
        print("[INFO] Menos de 2 componentes; gráfico 2D não criado.")

    # =====================================================
    # Metadados
    # =====================================================
    metadata = {
        "input_features_dir": str(FEATURES_DIR.resolve()),
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "classes": classes,
        "standardize_before_pca": STANDARDIZE_BEFORE_PCA,
        "pca_n_components": int(n_components),
        "used_feature_names_file": (FEATURES_DIR / "feature_names.txt").exists(),
    }

    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n[SUCESSO] Processo concluído.")
    print(f"[INFO] Saídas em: {OUTPUT_DIR.resolve()}")
    print("[INFO] Arquivos gerados:")
    print("  - class_mean_vectors_442.csv")
    print("  - class_min_vectors_442.csv")
    print("  - class_max_vectors_442.csv")
    print("  - class_counts_442.csv")
    print("  - pca_of_class_means_442.csv")
    print("  - pca_explained_variance_442.csv")
    print("  - pca_of_class_means_442_2d.png")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
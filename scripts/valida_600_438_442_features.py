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

DATASETS = {
    "600": Path("saida_features_600_comparacao"),
    "438": Path("saida_features_aa_dipep_v3_seed"),
    "442": Path("saida_features_aa_dipep_v4_442"),
}

RANDOM_STATE = 42
SAMPLE_SIZE = 200000
CLASSES_TO_KEEP = {1, 2, 3}

#BASE_OUTPUT_DIR = Path(".")
BASE_OUTPUT_DIR = Path("saida_valida_features")

# =========================================================
# AUXILIARES
# =========================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metadata(features_dir: Path) -> dict:
    meta_path = features_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json não encontrado em: {features_dir.resolve()}")

    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_memmap_info(features_dir: Path, meta: dict):
    x_path = meta.get("x_memmap_path")
    y_path = meta.get("y_memmap_path")
    shape_X = meta.get("shape_X")
    shape_y = meta.get("shape_y")

    if x_path is None or y_path is None or shape_X is None or shape_y is None:
        raise KeyError(
            f"metadata.json em {features_dir} precisa conter "
            f"x_memmap_path, y_memmap_path, shape_X e shape_y."
        )

    x_path = Path(x_path)
    y_path = Path(y_path)

    return x_path, y_path, tuple(shape_X), tuple(shape_y)


def load_memmaps(features_dir: Path):
    meta = load_metadata(features_dir)
    x_path, y_path, shape_X, shape_y = resolve_memmap_info(features_dir, meta)

    X = np.memmap(x_path, dtype=np.float32, mode="r", shape=shape_X)
    y = np.memmap(y_path, dtype=np.int32, mode="r", shape=shape_y)

    return X, y, meta


def sample_valid_indices(y_mem: np.memmap, sample_size: int, random_state: int) -> np.ndarray:
    y_array = np.asarray(y_mem)
    valid_idx = np.where(np.isin(y_array, list(CLASSES_TO_KEEP)))[0]

    if len(valid_idx) == 0:
        raise ValueError("Nenhuma linha restante após remover a classe 4.")

    rng = np.random.default_rng(random_state)

    if len(valid_idx) > sample_size:
        chosen = rng.choice(valid_idx, size=sample_size, replace=False)
        return np.asarray(chosen, dtype=np.int64)

    return np.asarray(valid_idx, dtype=np.int64)


def save_scatter_pca(X_pca: np.ndarray, y: np.ndarray, output_path: Path, title: str, evr: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))

    for cls in sorted(np.unique(y)):
        cls_mask = (y == cls)
        plt.scatter(
            X_pca[cls_mask, 0],
            X_pca[cls_mask, 1],
            s=6,
            alpha=0.35,
            label=f"Classe {cls}",
        )

    plt.title(title)
    plt.xlabel(f"PC1 ({evr[0] * 100:.2f}%)")
    plt.ylabel(f"PC2 ({evr[1] * 100:.2f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_class_means_plot(class_means_pca: pd.DataFrame, output_path: Path, title: str, evr: np.ndarray) -> None:
    plt.figure(figsize=(7, 6))

    plt.scatter(class_means_pca["PC1"], class_means_pca["PC2"], s=220)

    for _, row in class_means_pca.iterrows():
        plt.annotate(
            f"Classe {int(row['Class'])}",
            (row["PC1"], row["PC2"]),
            textcoords="offset points",
            xytext=(8, 8),
        )

    plt.title(title)
    plt.xlabel(f"PC1 ({evr[0] * 100:.2f}%)")
    plt.ylabel(f"PC2 ({evr[1] * 100:.2f}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =========================================================
# PROCESSAMENTO DE UM DATASET
# =========================================================

def process_dataset(dataset_name: str, features_dir: Path) -> None:
    print("\n" + "=" * 80)
    print(f"[INFO] Validando dataset {dataset_name}")
    print("=" * 80)

    output_dir = BASE_OUTPUT_DIR / f"saida_pca_classes_{dataset_name}"
    ensure_output_dir(output_dir)

    X_mem, y_mem, meta = load_memmaps(features_dir)

    print(f"[INFO] Shape original X: {X_mem.shape}")
    print(f"[INFO] Shape original y: {y_mem.shape}")

    idx = sample_valid_indices(y_mem, SAMPLE_SIZE, RANDOM_STATE)

    X_sample = np.asarray(X_mem[idx], dtype=np.float64)
    y_sample = np.asarray(y_mem[idx], dtype=np.int32)

    print(f"[INFO] Linhas usadas após remover classe 4: {len(y_sample):,}")
    print(f"[INFO] Classes presentes: {sorted(np.unique(y_sample).tolist())}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    # PCA do conjunto amostrado
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    scatter_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Class": y_sample,
    })
    scatter_df.to_csv(output_dir / f"pca_scatter_sampled_{dataset_name}.csv", index=False, encoding="utf-8")

    evr_df = pd.DataFrame({
        "Component": ["PC1", "PC2"],
        "ExplainedVarianceRatio": pca.explained_variance_ratio_,
        "ExplainedVariancePercent": pca.explained_variance_ratio_ * 100.0,
    })
    evr_df.to_csv(output_dir / f"pca_explained_variance_sampled_{dataset_name}.csv", index=False, encoding="utf-8")

    save_scatter_pca(
        X_pca,
        y_sample,
        output_dir / f"pca_scatter_sampled_{dataset_name}.png",
        title=f"PCA amostrado - {dataset_name} features (classes 1, 2 e 3)",
        evr=pca.explained_variance_ratio_,
    )

    # médias por classe
    class_rows = []
    class_means = []

    for cls in sorted(np.unique(y_sample)):
        cls_mean = X_scaled[y_sample == cls].mean(axis=0)
        class_means.append(cls_mean)
        class_rows.append(cls)

    class_means = np.asarray(class_means, dtype=np.float64)

    pca_means = PCA(n_components=2, random_state=RANDOM_STATE)
    class_means_pca = pca_means.fit_transform(class_means)

    class_means_df = pd.DataFrame({
        "Class": class_rows,
        "PC1": class_means_pca[:, 0],
        "PC2": class_means_pca[:, 1],
    })
    class_means_df.to_csv(output_dir / f"pca_of_class_means_{dataset_name}.csv", index=False, encoding="utf-8")

    evr_means_df = pd.DataFrame({
        "Component": ["PC1", "PC2"],
        "ExplainedVarianceRatio": pca_means.explained_variance_ratio_,
        "ExplainedVariancePercent": pca_means.explained_variance_ratio_ * 100.0,
    })
    evr_means_df.to_csv(output_dir / f"pca_explained_variance_{dataset_name}.csv", index=False, encoding="utf-8")

    save_class_means_plot(
        class_means_df,
        output_dir / f"pca_class_means_{dataset_name}.png",
        title=f"PCA das médias por classe - {dataset_name} features (sem classe 4)",
        evr=pca_means.explained_variance_ratio_,
    )

    info = {
        "dataset_name": dataset_name,
        "input_dir": str(features_dir.resolve()),
        "original_shape_X": list(X_mem.shape),
        "original_shape_y": list(y_mem.shape),
        "sample_size_after_class4_removal": int(len(y_sample)),
        "classes_used": sorted(list(CLASSES_TO_KEEP)),
    }

    with (output_dir / f"metadata_{dataset_name}.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[SUCESSO] Saídas salvas em: {output_dir.resolve()}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    for dataset_name, features_dir in DATASETS.items():
        process_dataset(dataset_name, features_dir)

    print("\n" + "=" * 90)
    print("[SUCESSO] Validação concluída para 600, 438 e 442 sem a classe 4.")
    print("=" * 90)


if __name__ == "__main__":
    main()
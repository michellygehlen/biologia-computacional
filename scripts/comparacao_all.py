import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURAÇÃO
# =========================================================

#datasets que não vou usar mais
#"420_features": Path("saida_features_aa_dipep"),
#"430_features": Path("saida_features_aa_dipep_v2"),

DATASETS = {
    "600_features": Path("saida_features_600_comparacao"),
    "438_seed_features": Path("saida_features_aa_dipep_v3_seed"),
    "442_seed_features": Path("saida_features_aa_dipep_v4_442"),
}

#experiments que não vou usar mais pq não melhorou em nada
#{"name": "600_PCA50_RF", "dataset": "600_features", "use_pca": True, "n_components": 50},
#{"name": "600_PCA60_RF", "dataset": "600_features", "use_pca": True, "n_components": 60},
#{"name": "420_RF", "dataset": "420_features", "use_pca": False, "n_components": None},
#{"name": "430_RF", "dataset": "430_features", "use_pca": False, "n_components": None},

EXPERIMENTS = [
    {"name": "600_RF", "dataset": "600_features", "use_pca": False, "n_components": None},
    {"name": "438_RF", "dataset": "438_seed_features", "use_pca": False, "n_components": None},
    {"name": "442_RF", "dataset": "442_seed_features", "use_pca": False, "n_components": None},
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

USE_SAMPLE = True
SAMPLE_SIZE = 200000

RF_PARAMS = {
    "n_estimators": 200,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "n_jobs": -1,
}


# =========================================================
# FUNÇÕES
# =========================================================

def load_dataset(output_dir: Path):
    with open(output_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    X = np.memmap(
        Path(meta["x_memmap_path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(meta["shape_X"]),
    )

    y = np.memmap(
        Path(meta["y_memmap_path"]),
        dtype=np.int32,
        mode="r",
        shape=tuple(meta["shape_y"]),
    )

    return X, y, meta


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


# =========================================================
# AMOSTRA E SPLIT COMUNS
# =========================================================

print("Carregando dataset-base para definir amostra comum...")
X_base, y_base, meta_base = load_dataset(DATASETS["420_features"])

n_base = len(y_base)
print(f"Total no dataset-base: {n_base:,}")

rng = np.random.default_rng(RANDOM_STATE)

if USE_SAMPLE and n_base > SAMPLE_SIZE:
    common_indices = rng.choice(n_base, size=SAMPLE_SIZE, replace=False)
else:
    common_indices = np.arange(n_base)

common_indices = np.asarray(common_indices, dtype=np.int64)
print(f"Amostra comum usada: {len(common_indices):,}\n")

y_sample_base = np.asarray(y_base[common_indices], dtype=np.int32)
sample_positions = np.arange(len(common_indices))

train_pos, test_pos = train_test_split(
    sample_positions,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_sample_base,
)

print(f"Treino comum: {len(train_pos):,}")
print(f"Teste comum : {len(test_pos):,}\n")


# =========================================================
# LOOP DE EXPERIMENTOS
# =========================================================

results = []

for exp in EXPERIMENTS:
    exp_name = exp["name"]
    ds_name = exp["dataset"]
    use_pca = exp["use_pca"]
    n_components = exp["n_components"]

    print("=" * 70)
    print(f"Experimento: {exp_name}")
    print("=" * 70)

    output_dir = DATASETS[ds_name]
    X_mem, y_mem, meta = load_dataset(output_dir)

    print(f"Dataset: {ds_name}")
    print(f"Shape completo X: {tuple(meta['shape_X'])}")
    print(f"Shape completo y: {tuple(meta['shape_y'])}")

    t0 = time.time()

    X_sample = np.asarray(X_mem[common_indices], dtype=np.float32)
    y_sample = np.asarray(y_mem[common_indices], dtype=np.int32)

    X_train = X_sample[train_pos]
    X_test = X_sample[test_pos]
    y_train = y_sample[train_pos]
    y_test = y_sample[test_pos]

    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")

    explained_variance = None

    if use_pca:
        print(f"Aplicando PCA com n_components={n_components} ...")
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)

        pca_t0 = time.time()
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        pca_time = time.time() - pca_t0

        explained_variance = float(np.sum(pca.explained_variance_ratio_))
        print(f"PCA concluído em {format_seconds(pca_time)}")
        print(f"Variância explicada total: {explained_variance:.4f}")
        print(f"Novo shape treino: {X_train.shape}")
        print(f"Novo shape teste : {X_test.shape}")

    print("Treinando Random Forest...")
    fit_t0 = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_t0

    print("Gerando previsões...")
    pred_t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - pred_t0

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    total_time = time.time() - t0

    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Tempo total: {format_seconds(total_time)}")

    results.append({
        "experiment": exp_name,
        "dataset": ds_name,
        "use_pca": use_pca,
        "n_components": n_components,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "explained_variance": explained_variance,
        "fit_time_sec": fit_time,
        "predict_time_sec": pred_time,
        "total_time_sec": total_time,
    })


# =========================================================
# TABELA FINAL
# =========================================================

print("\n" + "=" * 95)
print("COMPARAÇÃO FINAL")
print("=" * 95)

results_sorted = sorted(results, key=lambda x: x["macro_f1"], reverse=True)

header = (
    f"{'Experimento':<18} "
    f"{'Acc':<8} "
    f"{'MacroF1':<8} "
    f"{'WeightF1':<9} "
    f"{'PCA':<6} "
    f"{'n_comp':<8} "
    f"{'VarExp':<8} "
    f"{'Tempo':<10}"
)

print(header)
print("-" * len(header))

for r in results_sorted:
    var_exp_str = "-" if r["explained_variance"] is None else f"{r['explained_variance']:.4f}"
    n_comp_str = "-" if r["n_components"] is None else str(r["n_components"])
    pca_str = "sim" if r["use_pca"] else "não"

    print(
        f"{r['experiment']:<18} "
        f"{r['accuracy']:<8.4f} "
        f"{r['macro_f1']:<8.4f} "
        f"{r['weighted_f1']:<9.4f} "
        f"{pca_str:<6} "
        f"{n_comp_str:<8} "
        f"{var_exp_str:<8} "
        f"{format_seconds(r['total_time_sec']):<10}"
    )

if results_sorted:
    best = results_sorted[0]
    print("\nMelhor experimento por Macro F1:")
    print(
        f"{best['experiment']} | "
        f"Macro F1 = {best['macro_f1']:.4f} | "
        f"Accuracy = {best['accuracy']:.4f}"
    )
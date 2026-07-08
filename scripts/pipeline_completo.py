from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# =========================================================
# CONFIGURAÇÕES
# =========================================================

DATASETS = {
    "600_features": Path("saida_features_600_comparacao"),
    "438_seed_features": Path("saida_features_aa_dipep_v3_seed"),
    "442_seed_features": Path("saida_features_aa_dipep_v4_442"),
}

EXPERIMENTS = [
    {"name": "600_BASE", "dataset": "600_features", "use_pca": False, "n_components": None},
    {"name": "438_BASE", "dataset": "438_seed_features", "use_pca": False, "n_components": None},
    {"name": "442_BASE", "dataset": "442_seed_features", "use_pca": False, "n_components": None},

    # Se quiser reativar variantes com PCA:
    # {"name": "600_PCA50", "dataset": "600_features", "use_pca": True, "n_components": 50},
    # {"name": "600_PCA60", "dataset": "600_features", "use_pca": True, "n_components": 60},
]

OUTPUT_DIR = Path("saida_comparacao_all_probabilidades_modelos_completos")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Para acelerar.
# Se quiser usar tudo, coloque False.
USE_SAMPLE = False
SAMPLE_SIZE = 200000

# Cenários do paper
SCENARIOS = [
    {"name": "multiclasse_4classes", "kind": "multiclass", "classes": [1, 2, 3, 4]},
    {"name": "multiclasse_sem_classe4", "kind": "multiclass", "classes": [1, 2, 3]},
    {"name": "binario_classe1_vs_resto", "kind": "binary", "positive_class": 1},
    {"name": "binario_classe2_vs_resto", "kind": "binary", "positive_class": 2},
    {"name": "binario_classe3_vs_resto", "kind": "binary", "positive_class": 3},
]

SAVE_PREDICTIONS = True
SAVE_PLOTS = True
PRINT_REPORTS = True


# =========================================================
# DIRETÓRIOS
# =========================================================

REPORTS_DIR = OUTPUT_DIR / "reports"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PLOTS_DIR = OUTPUT_DIR / "plots"


# =========================================================
# MODELOS
# =========================================================

def get_models() -> dict[str, object]:
    return {
        "MLP_FULL": MLPClassifier(
            hidden_layer_sizes=(29, 13, 5),
            activation="relu",
            solver="adam",
            random_state=RANDOM_STATE,
            max_iter=600,
        ),
        "MLP_HALF": MLPClassifier(
            hidden_layer_sizes=(14, 6, 3),
            activation="relu",
            solver="adam",
            random_state=RANDOM_STATE,
            max_iter=600,
        ),
        "RF_FULL": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "RF_HALF": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# =========================================================
# AUXILIARES
# =========================================================

def ensure_dirs() -> None:
    for d in [OUTPUT_DIR, REPORTS_DIR, PREDICTIONS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


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

    headers_path = output_dir / "headers.txt"
    if headers_path.exists():
        with open(headers_path, "r", encoding="utf-8") as f:
            headers = np.asarray([line.rstrip("\n") for line in f], dtype=object)
    else:
        headers = np.asarray([f"row_{i}" for i in range(len(y))], dtype=object)

    return X, y, headers, meta


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def class_distribution(y_array: np.ndarray) -> dict[int, tuple[int, float]]:
    values, counts = np.unique(y_array, return_counts=True)
    total = counts.sum()
    return {int(v): (int(c), float(c / total)) for v, c in zip(values, counts)}


def print_distribution(title: str, y_array: np.ndarray) -> None:
    dist = class_distribution(y_array)
    print(title)
    for cls in sorted(dist):
        count, frac = dist[cls]
        print(f"  Classe {cls}: {count:,} ({frac * 100:.2f}%)")


def interpret_gap(train_acc: float, test_acc: float, train_f1: float, test_f1: float) -> str:
    gap_acc = train_acc - test_acc
    gap_f1 = train_f1 - test_f1

    if gap_acc < 0.03 and gap_f1 < 0.03:
        return "Boa generalização"
    if gap_acc < 0.08 and gap_f1 < 0.08:
        return "Leve overfitting"
    if gap_acc < 0.15 and gap_f1 < 0.15:
        return "Overfitting moderado"
    return "Overfitting forte"


def get_base_dataset_key() -> str:
    # Usa o primeiro dataset ativo como referência para a amostra comum.
    return next(iter(DATASETS.keys()))


def build_common_sample():
    base_key = get_base_dataset_key()
    print(f"Carregando dataset-base para definir amostra comum: {base_key}")

    X_base, y_base, headers_base, _ = load_dataset(DATASETS[base_key])

    n_total = len(y_base)
    print(f"Total no dataset-base: {n_total:,}")

    rng = np.random.default_rng(RANDOM_STATE)

    if USE_SAMPLE and n_total > SAMPLE_SIZE:
        common_indices = rng.choice(n_total, size=SAMPLE_SIZE, replace=False)
    else:
        common_indices = np.arange(n_total)

    common_indices = np.asarray(common_indices, dtype=np.int64)

    y_sample_base = np.asarray(y_base[common_indices], dtype=np.int32)
    headers_sample_base = np.asarray(headers_base[common_indices], dtype=object)

    print(f"Amostra comum usada: {len(common_indices):,}")
    print_distribution("Distribuição da amostra-base:", y_sample_base)
    print()

    return common_indices, y_sample_base, headers_sample_base


def build_binary_balanced(
    X: np.ndarray,
    y: np.ndarray,
    headers: np.ndarray,
    positive_class: int,
):
    """
    Gera problema binário 1:1 balanceado.
    Mantém apenas classes 1, 2 e 3 e compara uma classe positiva contra o resto.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    idx_pos = np.where(y == positive_class)[0]
    idx_neg_pool = np.where(y != positive_class)[0]

    if len(idx_pos) == 0:
        raise ValueError(f"Nenhuma amostra encontrada para a classe positiva {positive_class}.")

    if len(idx_neg_pool) < len(idx_pos):
        raise ValueError(
            f"Negativos insuficientes para balancear a classe {positive_class}: "
            f"positivos={len(idx_pos)}, negativos={len(idx_neg_pool)}"
        )

    idx_neg = rng.choice(idx_neg_pool, size=len(idx_pos), replace=False)

    idx_final = np.concatenate([idx_pos, idx_neg])
    rng.shuffle(idx_final)

    X_bin = X[idx_final].copy()
    y_bin = (y[idx_final] == positive_class).astype(np.int32)
    h_bin = headers[idx_final].copy()

    return X_bin, y_bin, h_bin


def apply_scenario(X: np.ndarray, y: np.ndarray, headers: np.ndarray, scenario: dict):
    if scenario["kind"] == "multiclass":
        mask = np.isin(y, scenario["classes"])
        return X[mask].copy(), y[mask].copy(), headers[mask].copy()

    if scenario["kind"] == "binary":
        mask_123 = np.isin(y, [1, 2, 3])
        X_123 = X[mask_123]
        y_123 = y[mask_123]
        h_123 = headers[mask_123]
        return build_binary_balanced(X_123, y_123, h_123, scenario["positive_class"])

    raise ValueError(f"Cenário desconhecido: {scenario}")


def maybe_apply_pca(X_train, X_test, use_pca: bool, n_components: int | None):
    if not use_pca:
        return X_train, X_test, None, 0.0

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)

    t0 = time.time()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_time = time.time() - t0

    explained_variance = float(np.sum(pca.explained_variance_ratio_))
    return X_train_pca, X_test_pca, explained_variance, pca_time


def save_predictions_csv(
    headers: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_classes: np.ndarray,
    out_csv: Path,
) -> None:
    data = {
        "Header": headers,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence_max": y_proba.max(axis=1),
        "correct": (y_true == y_pred).astype(int),
    }

    for i, cls in enumerate(model_classes):
        data[f"prob_class_{int(cls)}"] = y_proba[:, i]

    pd.DataFrame(data).to_csv(out_csv, index=False, encoding="utf-8")


def save_confusion_matrix_csv(cm: np.ndarray, classes: list[int], out_csv: Path) -> None:
    df_cm = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )
    df_cm.to_csv(out_csv, encoding="utf-8")


def save_confusion_matrix_plot(cm: np.ndarray, classes: list[int], title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confidence_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    confidences = y_proba.max(axis=1)
    correct_mask = y_true == y_pred
    wrong_mask = ~correct_mask

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0.0, 1.0, 21)

    if correct_mask.any():
        ax.hist(confidences[correct_mask], bins=bins, alpha=0.7, label="Acertos")
    if wrong_mask.any():
        ax.hist(confidences[wrong_mask], bins=bins, alpha=0.7, label="Erros")

    ax.set_xlabel("Confiança da previsão (probabilidade máxima)")
    ax.set_ylabel("Frequência")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_multiclass_probability_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_classes: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)

    plotted = False
    for cls in model_classes:
        mask = y_true == cls
        if mask.any():
            col = np.where(model_classes == cls)[0][0]
            ax.hist(
                y_proba[mask, col],
                bins=bins,
                alpha=0.5,
                label=f"Classe real {int(cls)}",
            )
            plotted = True

    if plotted:
        ax.set_xlabel("Probabilidade atribuída à classe correta")
        ax.set_ylabel("Frequência")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")

    plt.close(fig)


def save_binary_probability_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)

    ax.hist(y_proba[y_true == 0, 1], bins=bins, alpha=0.7, label="Classe 0 (resto)")
    ax.hist(y_proba[y_true == 1, 1], bins=bins, alpha=0.7, label="Classe 1 (positiva)")

    ax.set_xlabel("Probabilidade da classe positiva")
    ax.set_ylabel("Frequência")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_calibration_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_classes: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")

    plotted = False

    if y_proba.shape[1] == 2 and set(np.unique(y_true)) <= {0, 1}:
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_proba[:, 1],
            n_bins=10,
            strategy="uniform",
        )
        ax.plot(prob_pred, prob_true, marker="o", label="Classe positiva")
        plotted = True
    else:
        for cls in model_classes:
            y_bin = (y_true == cls).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue

            col = np.where(model_classes == cls)[0][0]
            prob_true, prob_pred = calibration_curve(
                y_bin,
                y_proba[:, col],
                n_bins=10,
                strategy="uniform",
            )
            ax.plot(prob_pred, prob_true, marker="o", label=f"Classe {int(cls)}")
            plotted = True

    if plotted:
        ax.set_xlabel("Probabilidade média prevista")
        ax.set_ylabel("Fração observada de positivos")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")

    plt.close(fig)


def extract_class_f1(report_dict: dict, label: str):
    return report_dict.get(label, {}).get("f1-score")


# =========================================================
# EXPERIMENTO
# =========================================================

def run_experiment(exp: dict, common_indices: np.ndarray):
    exp_name = exp["name"]
    ds_name = exp["dataset"]
    use_pca = exp["use_pca"]
    n_components = exp["n_components"]

    print("=" * 120)
    print(f"Experimento base: {exp_name}")
    print("=" * 120)

    X_mem, y_mem, headers_mem, meta = load_dataset(DATASETS[ds_name])

    X_sample = np.asarray(X_mem[common_indices], dtype=np.float32)
    y_sample = np.asarray(y_mem[common_indices], dtype=np.int32)
    headers_sample = np.asarray(headers_mem[common_indices], dtype=object)

    print(f"Dataset: {ds_name}")
    print(f"Shape completo X: {tuple(meta['shape_X'])}")
    print(f"Shape completo y: {tuple(meta['shape_y'])}")
    print(f"Shape da amostra comum: {X_sample.shape}")
    print_distribution("Distribuição na amostra comum:", y_sample)
    print()

    results = []

    for scenario in SCENARIOS:
        scenario_name = scenario["name"]

        print("-" * 120)
        print(f"Cenário: {scenario_name}")
        print("-" * 120)

        X_s, y_s, h_s = apply_scenario(X_sample, y_sample, headers_sample, scenario)

        print(f"Shape após cenário: {X_s.shape}")
        print_distribution("Distribuição após cenário:", y_s)

        X_train, X_test, y_train, y_test, h_train, h_test = train_test_split(
            X_s,
            y_s,
            h_s,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_s,
        )

        print(f"X_train: {X_train.shape}")
        print(f"X_test : {X_test.shape}")

        explained_variance = None
        pca_time = 0.0

        if use_pca:
            X_train, X_test, explained_variance, pca_time = maybe_apply_pca(
                X_train,
                X_test,
                use_pca,
                n_components,
            )
            print(
                f"PCA concluído em {format_seconds(pca_time)} | "
                f"Variância explicada total: {explained_variance:.4f}"
            )

        for model_name, model in get_models().items():
            tag = f"{exp_name}__{scenario_name}__{model_name}"

            print("." * 100)
            print(f"MODELO: {model_name}")
            print("." * 100)

            fit_t0 = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - fit_t0

            pred_t0 = time.time()
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(X_test)
            else:
                raise RuntimeError(f"O modelo {model_name} não possui predict_proba().")

            pred_time = time.time() - pred_t0

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            train_macro_f1 = f1_score(y_train, y_train_pred, average="macro")
            test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")

            train_weighted_f1 = f1_score(y_train, y_train_pred, average="weighted")
            test_weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")

            gap_acc = train_acc - test_acc
            gap_macro_f1 = train_macro_f1 - test_macro_f1
            interpretation = interpret_gap(train_acc, test_acc, train_macro_f1, test_macro_f1)

            total_time = fit_time + pred_time + pca_time
            model_classes = model.classes_

            cm = confusion_matrix(y_test, y_test_pred, labels=model_classes)
            report_dict = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
            report_txt = classification_report(y_test, y_test_pred, digits=4)

            print(f"Train Accuracy     : {train_acc:.4f}")
            print(f"Test Accuracy      : {test_acc:.4f}")
            print(f"Gap Accuracy       : {gap_acc:.4f}")
            print(f"Train Macro F1     : {train_macro_f1:.4f}")
            print(f"Test Macro F1      : {test_macro_f1:.4f}")
            print(f"Gap Macro F1       : {gap_macro_f1:.4f}")
            print(f"Weighted F1 (teste): {test_weighted_f1:.4f}")
            print(f"Interpretação      : {interpretation}")
            print(f"Tempo total        : {format_seconds(total_time)}")

            if PRINT_REPORTS:
                print("Classification report (teste):")
                print(report_txt)
                print("Confusion matrix (teste):")
                print(cm)

            report_path = REPORTS_DIR / f"{tag}_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Experimento base: {exp_name}\n")
                f.write(f"Cenário: {scenario_name}\n")
                f.write(f"Modelo: {model_name}\n")
                f.write(f"Dataset: {ds_name}\n\n")
                f.write(f"Train Accuracy: {train_acc:.6f}\n")
                f.write(f"Test Accuracy: {test_acc:.6f}\n")
                f.write(f"Gap Accuracy: {gap_acc:.6f}\n")
                f.write(f"Train Macro F1: {train_macro_f1:.6f}\n")
                f.write(f"Test Macro F1: {test_macro_f1:.6f}\n")
                f.write(f"Gap Macro F1: {gap_macro_f1:.6f}\n")
                f.write(f"Train Weighted F1: {train_weighted_f1:.6f}\n")
                f.write(f"Test Weighted F1: {test_weighted_f1:.6f}\n")
                f.write(f"Interpretação: {interpretation}\n")
                f.write(f"Fit time: {fit_time:.6f}\n")
                f.write(f"Predict time: {pred_time:.6f}\n")
                f.write(f"Total time: {total_time:.6f}\n")
                if explained_variance is not None:
                    f.write(f"Explained variance: {explained_variance:.6f}\n")
                f.write("\nClassification report (teste):\n")
                f.write(report_txt)
                f.write("\nConfusion matrix (teste):\n")
                f.write(str(cm))

            save_confusion_matrix_csv(cm, list(model_classes), PLOTS_DIR / f"{tag}_confusion_matrix.csv")
            if SAVE_PLOTS:
                save_confusion_matrix_plot(
                    cm,
                    list(model_classes),
                    f"Matriz de confusão - {tag}",
                    PLOTS_DIR / f"{tag}_confusion_matrix.png",
                )

            if SAVE_PREDICTIONS:
                save_predictions_csv(
                    h_test,
                    y_test,
                    y_test_pred,
                    y_test_proba,
                    model_classes,
                    PREDICTIONS_DIR / f"{tag}_predictions.csv",
                )

            if SAVE_PLOTS:
                save_confidence_histogram(
                    y_true=y_test,
                    y_pred=y_test_pred,
                    y_proba=y_test_proba,
                    title=f"Confiança: acertos vs erros - {tag}",
                    out_png=PLOTS_DIR / f"{tag}_confidence_correct_vs_error.png",
                )

                if scenario["kind"] == "binary":
                    save_binary_probability_plot(
                        y_true=y_test,
                        y_proba=y_test_proba,
                        title=f"Probabilidade da classe positiva - {tag}",
                        out_png=PLOTS_DIR / f"{tag}_positive_probability_hist.png",
                    )
                else:
                    save_multiclass_probability_plot(
                        y_true=y_test,
                        y_proba=y_test_proba,
                        model_classes=model_classes,
                        title=f"Probabilidade da classe correta por classe real - {tag}",
                        out_png=PLOTS_DIR / f"{tag}_probability_by_true_class.png",
                    )

                save_calibration_plot(
                    y_true=y_test,
                    y_proba=y_test_proba,
                    model_classes=model_classes,
                    title=f"Calibration plot - {tag}",
                    out_png=PLOTS_DIR / f"{tag}_calibration.png",
                )

            result_row = {
                "experiment": exp_name,
                "dataset": ds_name,
                "scenario": scenario_name,
                "problem_type": scenario["kind"],
                "model": model_name,
                "use_pca": use_pca,
                "n_components": n_components,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "gap_accuracy": gap_acc,
                "train_macro_f1": train_macro_f1,
                "test_macro_f1": test_macro_f1,
                "gap_macro_f1": gap_macro_f1,
                "train_weighted_f1": train_weighted_f1,
                "test_weighted_f1": test_weighted_f1,
                "interpretation": interpretation,
                "explained_variance": explained_variance,
                "fit_time_sec": fit_time,
                "predict_time_sec": pred_time,
                "total_time_sec": total_time,
                "n_total_scenario": len(y_s),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "class_0_f1": extract_class_f1(report_dict, "0"),
                "class_1_f1": extract_class_f1(report_dict, "1"),
                "class_2_f1": extract_class_f1(report_dict, "2"),
                "class_3_f1": extract_class_f1(report_dict, "3"),
                "class_4_f1": extract_class_f1(report_dict, "4"),
            }

            results.append(result_row)

    return results


# =========================================================
# SUMÁRIOS
# =========================================================

def save_summary_tables(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_DIR / "summary_all_scenarios_all_models.csv", index=False, encoding="utf-8")

    df_sorted = df.sort_values(
        by=["scenario", "test_macro_f1", "test_accuracy"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    df_sorted.to_csv(
        OUTPUT_DIR / "summary_all_scenarios_all_models_sorted.csv",
        index=False,
        encoding="utf-8",
    )

    paper_cols = [
        "experiment",
        "dataset",
        "scenario",
        "problem_type",
        "model",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "gap_accuracy",
        "gap_macro_f1",
        "interpretation",
        "class_0_f1",
        "class_1_f1",
        "class_2_f1",
        "class_3_f1",
        "class_4_f1",
        "n_total_scenario",
    ]
    df_paper = df_sorted[paper_cols].copy()
    df_paper.to_csv(OUTPUT_DIR / "summary_for_paper.csv", index=False, encoding="utf-8")

    df_pivot = df.pivot_table(
        index=["dataset", "scenario", "model"],
        values=[
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            "gap_accuracy",
            "gap_macro_f1",
        ],
        aggfunc="mean",
    ).reset_index()
    df_pivot.to_csv(OUTPUT_DIR / "summary_pivot_dataset_scenario_model.csv", index=False, encoding="utf-8")

    best_by_scenario = (
        df.sort_values(by=["scenario", "test_macro_f1", "test_accuracy"], ascending=[True, False, False])
          .groupby("scenario", as_index=False)
          .first()
    )
    best_by_scenario.to_csv(OUTPUT_DIR / "best_model_per_scenario.csv", index=False, encoding="utf-8")

    best_by_dataset_scenario = (
        df.sort_values(by=["dataset", "scenario", "test_macro_f1", "test_accuracy"], ascending=[True, True, False, False])
          .groupby(["dataset", "scenario"], as_index=False)
          .first()
    )
    best_by_dataset_scenario.to_csv(
        OUTPUT_DIR / "best_model_per_dataset_and_scenario.csv",
        index=False,
        encoding="utf-8",
    )


def print_final_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 160)
    print("COMPARAÇÃO FINAL")
    print("=" * 160)

    df_sorted = df.sort_values(by=["scenario", "test_macro_f1"], ascending=[True, False])

    header = (
        f"{'Experimento':<12} "
        f"{'Cenário':<28} "
        f"{'Modelo':<10} "
        f"{'Acc':<8} "
        f"{'MacroF1':<8} "
        f"{'WeightF1':<9} "
        f"{'GapAcc':<8} "
        f"{'GapF1':<8} "
        f"{'Tempo':<10}"
    )
    print(header)
    print("-" * len(header))

    for _, r in df_sorted.iterrows():
        print(
            f"{str(r['experiment']):<12} "
            f"{str(r['scenario']):<28} "
            f"{str(r['model']):<10} "
            f"{float(r['test_accuracy']):<8.4f} "
            f"{float(r['test_macro_f1']):<8.4f} "
            f"{float(r['test_weighted_f1']):<9.4f} "
            f"{float(r['gap_accuracy']):<8.4f} "
            f"{float(r['gap_macro_f1']):<8.4f} "
            f"{format_seconds(float(r['total_time_sec'])):<10}"
        )

    print("\nMelhor combinação por cenário:")
    for scenario_name, group in df.groupby("scenario"):
        best = group.sort_values(by=["test_macro_f1", "test_accuracy"], ascending=[False, False]).iloc[0]
        print(
            f" - {scenario_name}: "
            f"{best['experiment']} | {best['model']} | "
            f"Macro F1 = {best['test_macro_f1']:.4f} | "
            f"Accuracy = {best['test_accuracy']:.4f} | "
            f"Gap F1 = {best['gap_macro_f1']:.4f}"
        )


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dirs()

    common_indices, _, _ = build_common_sample()

    all_results = []

    for exp in EXPERIMENTS:
        results = run_experiment(exp, common_indices)
        all_results.extend(results)

    df_final = pd.DataFrame(all_results)
    save_summary_tables(df_final)
    print_final_summary(df_final)

    print("\n[INFO] Arquivos gerados em:")
    print(f" - {OUTPUT_DIR}")
    print("   - summary_all_scenarios_all_models.csv")
    print("   - summary_all_scenarios_all_models_sorted.csv")
    print("   - summary_for_paper.csv")
    print("   - summary_pivot_dataset_scenario_model.csv")
    print("   - best_model_per_scenario.csv")
    print("   - best_model_per_dataset_and_scenario.csv")
    print("   - reports/*.txt")
    print("   - predictions/*_predictions.csv")
    print("   - plots/*_confusion_matrix.csv")
    print("   - plots/*_confusion_matrix.png")
    print("   - plots/*_confidence_correct_vs_error.png")
    print("   - plots/*_probability_by_true_class.png ou *_positive_probability_hist.png")
    print("   - plots/*_calibration.png")


if __name__ == "__main__":
    main()
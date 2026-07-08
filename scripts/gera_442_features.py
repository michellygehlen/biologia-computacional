from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURAÇÕES
# =========================================================

#origem para treinamento do modelo com 6 milhões de registros <--- ainda não gerei 31/03/26
#INPUT_CSV = Path("result_db/origem_600_features.csv")
#OUTPUT_DIR = Path("saida_features_aa_dipep_v4_442_6milhoes")

#origem para treinamento do modelo com 500 mil linhas de amostra
#INPUT_CSV = Path("result_db/resultado_debug_500k.csv")
#OUTPUT_DIR = Path("saida_features_aa_dipep_v4_442")

INPUT_CSV = Path("result_db/resultado_debug_500k.csv")
OUTPUT_DIR = Path("saida_features_aa_dipep_v4_442")

# Colunas do CSV
SEQ_COL = "Sequence"
LABEL_COL = "GroupID"
HEADER_COL = "FastaHeader"

# Separador do arquivo
CSV_SEPARATOR = ","
CSV_ENCODING = "utf-8"

# Aminoácidos padrão
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Grupos físico-químicos
PHYSICOCHEM_GROUPS = {
    "hydrophobic": set("AVILMGP"),
    "polar_uncharged": set("STNQCY"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "aromatic": set("FWY"),
    "sulfur": set("CM"),
    "special": set("PG"),
}
PHYSICOCHEM_GROUP_NAMES = list(PHYSICOCHEM_GROUPS.keys())

# Seed 2x1x2
WINDOW_SIZE = 5
DIPEPTIDE_DIM = 400

SEED_FEATURE_NAMES = [
    "SEED_COUNT",
    "SEED_DENSITY",
    "SEED_UNIQUE_COUNT",
    "SEED_UNIQUE_RATIO",
    "SEED_ENTROPY",
    "SEED_TOP1_RATIO",
    "SEED_TOP3_RATIO",
    "SEED_SINGLETON_RATIO",
]
SEED_STATS_DIM = len(SEED_FEATURE_NAMES)

# Novas features ProtParam-like
PROTPARAM_FEATURE_NAMES = [
    "MOL_WEIGHT",
    "PI_THEORETICAL",
    "GRAVY",
    "INSTABILITY_INDEX",
]
PROTPARAM_DIM = len(PROTPARAM_FEATURE_NAMES)

# Dimensões
AA_DIM = 20
EXTRA_DIM = 3 + len(PHYSICOCHEM_GROUP_NAMES)   # length, log_length, entropy + grupos
TOTAL_DIM = AA_DIM + DIPEPTIDE_DIM + EXTRA_DIM + SEED_STATS_DIM + PROTPARAM_DIM

# Processamento
CHUNK_SIZE = 50000
FLUSH_EVERY_CHUNKS = 5
REPORT_EVERY_ROWS = 100000

# Controle de saída
WRITE_FEATURE_NAMES_TXT = True
DELETE_OLD_OUTPUT_IF_NO_CHECKPOINT = True

# Normalização
NORMALIZE_AA_FREQ = True
NORMALIZE_DIPEPTIDE_FREQ = True

# Escala do comprimento
USE_RAW_LENGTH = True
USE_LOG_LENGTH = True

# Tipos
X_DTYPE = np.float32
Y_DTYPE = np.int32


# =========================================================
# CAMINHOS DE SAÍDA
# =========================================================

X_DAT = OUTPUT_DIR / "X_features_v4_442_float32.dat"
Y_DAT = OUTPUT_DIR / "y_labels_int32.dat"
HEADERS_TXT = OUTPUT_DIR / "headers.txt"
FEATURE_NAMES_TXT = OUTPUT_DIR / "feature_names.txt"
CHECKPOINT_JSON = OUTPUT_DIR / "checkpoint.json"
METADATA_JSON = OUTPUT_DIR / "metadata.json"


# =========================================================
# DADOS BIOQUÍMICOS
# =========================================================

# massas residuais aproximadas (após perda de H2O na ligação peptídica)
AA_MASS = {
    "A": 71.0788, "C": 103.1388, "D": 115.0886, "E": 129.1155, "F": 147.1766,
    "G": 57.0519, "H": 137.1411, "I": 113.1594, "K": 128.1741, "L": 113.1594,
    "M": 131.1926, "N": 114.1038, "P": 97.1167, "Q": 128.1307, "R": 156.1875,
    "S": 87.0782, "T": 101.1051, "V": 99.1326, "W": 186.2132, "Y": 163.1760,
}
WATER_MASS = 18.01528

# escala Kyte-Doolittle para GRAVY
HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# pKa aproximados
PKA_NTERM = 9.69
PKA_CTERM = 2.34
PKA_SIDE = {
    "C": 8.33,
    "D": 3.86,
    "E": 4.25,
    "H": 6.00,
    "K": 10.53,
    "R": 12.48,
    "Y": 10.07,
}

# pesos aproximados para instability-like
# regra simples baseada em variação de hidropatia + presença de resíduos especiais/carregados
# é uma aproximação consistente para experimento incremental
def build_instability_pair_weights():
    weights = {}
    for a1 in AMINO_ACIDS:
        for a2 in AMINO_ACIDS:
            score = abs(HYDROPATHY[a1] - HYDROPATHY[a2])
            if a1 in "PG" or a2 in "PG":
                score += 0.8
            if a1 in "DEKRH" or a2 in "DEKRH":
                score += 0.5
            weights[a1 + a2] = score
    return weights

INSTABILITY_PAIR_WEIGHT = build_instability_pair_weights()


# =========================================================
# AUXILIARES
# =========================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_file_if_exists(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def remove_output_files() -> None:
    for path in [
        X_DAT,
        Y_DAT,
        HEADERS_TXT,
        FEATURE_NAMES_TXT,
        CHECKPOINT_JSON,
        METADATA_JSON,
    ]:
        remove_file_if_exists(path)


def checkpoint_exists() -> bool:
    return CHECKPOINT_JSON.exists()


def partial_output_exists() -> bool:
    return any(path.exists() for path in [X_DAT, Y_DAT, HEADERS_TXT, FEATURE_NAMES_TXT, METADATA_JSON])


def save_metadata(metadata: dict) -> None:
    with METADATA_JSON.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_checkpoint(processed_rows: int) -> None:
    with CHECKPOINT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"processed_rows": processed_rows}, f)


def load_checkpoint() -> int:
    if not CHECKPOINT_JSON.exists():
        return 0
    with CHECKPOINT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("processed_rows", 0))


def finalize_success() -> None:
    remove_file_if_exists(CHECKPOINT_JSON)
    print("[INFO] Checkpoint removido após conclusão com sucesso.")


def prepare_run() -> int:
    ensure_output_dir(OUTPUT_DIR)

    has_checkpoint = checkpoint_exists()
    has_partial = partial_output_exists()

    if has_checkpoint:
        processed_rows = load_checkpoint()
        print(f"[INFO] Continuando a partir de {processed_rows:,} linhas.")
        return processed_rows

    if has_partial:
        if DELETE_OLD_OUTPUT_IF_NO_CHECKPOINT:
            print("[INFO] Saídas antigas encontradas sem checkpoint. Apagando para começar do zero...")
            remove_output_files()
        else:
            raise RuntimeError(
                "Foram encontrados arquivos antigos sem checkpoint. "
                "Apague manualmente ou altere DELETE_OLD_OUTPUT_IF_NO_CHECKPOINT para True."
            )

    return 0


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding=CSV_ENCODING, errors="ignore") as f:
        total = sum(1 for _ in f) - 1
    return max(total, 0)


def build_feature_names() -> list[str]:
    aa_cols = [f"AA_{aa}" for aa in AMINO_ACIDS]
    dipep_cols = [f"DIPEP_{a1}{a2}" for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS]

    extra_cols = []
    if USE_RAW_LENGTH:
        extra_cols.append("SEQ_LENGTH")
    if USE_LOG_LENGTH:
        extra_cols.append("SEQ_LOG_LENGTH")
    extra_cols.append("AA_ENTROPY")
    extra_cols.extend([f"PROP_{name.upper()}" for name in PHYSICOCHEM_GROUP_NAMES])

    return aa_cols + dipep_cols + extra_cols + SEED_FEATURE_NAMES + PROTPARAM_FEATURE_NAMES


def save_feature_names() -> None:
    if not WRITE_FEATURE_NAMES_TXT:
        return

    cols = build_feature_names()
    with FEATURE_NAMES_TXT.open("w", encoding="utf-8", newline="\n") as f:
        for col in cols:
            f.write(col + "\n")


def prepare_output_memmaps(total_rows: int) -> tuple[np.memmap, np.memmap]:
    print("[INFO] Criando memmaps de saída...")
    X_mem = np.memmap(
        X_DAT,
        dtype=X_DTYPE,
        mode="w+",
        shape=(total_rows, TOTAL_DIM),
    )
    y_mem = np.memmap(
        Y_DAT,
        dtype=Y_DTYPE,
        mode="w+",
        shape=(total_rows,),
    )
    return X_mem, y_mem


def load_existing_memmaps(total_rows: int) -> tuple[np.memmap, np.memmap]:
    X_mem = np.memmap(
        X_DAT,
        dtype=X_DTYPE,
        mode="r+",
        shape=(total_rows, TOTAL_DIM),
    )
    y_mem = np.memmap(
        Y_DAT,
        dtype=Y_DTYPE,
        mode="r+",
        shape=(total_rows,),
    )
    return X_mem, y_mem


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def clean_sequence(seq: str) -> str:
    return str(seq).strip().upper()


def shannon_entropy_from_freq(freqs: np.ndarray) -> float:
    nonzero = freqs[freqs > 0]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))


def dipeptide_index(a1: str, a2: str) -> int | None:
    idx1 = AA_TO_IDX.get(a1)
    idx2 = AA_TO_IDX.get(a2)
    if idx1 is None or idx2 is None:
        return None
    return idx1 * 20 + idx2


def sequence_seed_indices_2x1x2(seq: str) -> np.ndarray:
    seq = clean_sequence(seq)
    n = len(seq)

    if n < WINDOW_SIZE:
        return np.empty(0, dtype=np.int32)

    indices = []

    for pos in range(n - WINDOW_SIZE + 1):
        a = seq[pos]
        b = seq[pos + 1]
        c = seq[pos + 3]
        d = seq[pos + 4]

        left = dipeptide_index(a, b)
        right = dipeptide_index(c, d)

        if left is None or right is None:
            continue

        idx = left * DIPEPTIDE_DIM + right
        indices.append(idx)

    if not indices:
        return np.empty(0, dtype=np.int32)

    return np.asarray(indices, dtype=np.int32)


def seed_summary_features(seq: str) -> np.ndarray:
    seq = clean_sequence(seq)
    n = len(seq)

    max_possible_windows = max(n - WINDOW_SIZE + 1, 0)
    seed_indices = sequence_seed_indices_2x1x2(seq)

    seed_count = int(seed_indices.size)

    if seed_count == 0:
        return np.zeros(SEED_STATS_DIM, dtype=np.float32)

    unique_idx, counts = np.unique(seed_indices, return_counts=True)

    seed_unique_count = int(unique_idx.size)
    seed_density = seed_count / max_possible_windows if max_possible_windows > 0 else 0.0
    seed_unique_ratio = seed_unique_count / seed_count if seed_count > 0 else 0.0

    probs = counts.astype(np.float64) / seed_count
    seed_entropy = float(-np.sum(probs * np.log2(probs))) if probs.size > 0 else 0.0

    counts_sorted = np.sort(counts)[::-1]
    seed_top1_ratio = float(counts_sorted[0] / seed_count) if counts_sorted.size >= 1 else 0.0
    seed_top3_ratio = float(counts_sorted[:3].sum() / seed_count) if counts_sorted.size >= 1 else 0.0

    singleton_count = int(np.sum(counts == 1))
    seed_singleton_ratio = singleton_count / seed_unique_count if seed_unique_count > 0 else 0.0

    return np.asarray([
        float(seed_count),
        float(seed_density),
        float(seed_unique_count),
        float(seed_unique_ratio),
        float(seed_entropy),
        float(seed_top1_ratio),
        float(seed_top3_ratio),
        float(seed_singleton_ratio),
    ], dtype=np.float32)


def molecular_weight(seq: str) -> float:
    valid = [aa for aa in seq if aa in AA_MASS]
    if not valid:
        return 0.0
    return float(sum(AA_MASS[aa] for aa in valid) + WATER_MASS)


def gravy(seq: str) -> float:
    valid = [aa for aa in seq if aa in HYDROPATHY]
    if not valid:
        return 0.0
    return float(sum(HYDROPATHY[aa] for aa in valid) / len(valid))


def net_charge_at_pH(seq: str, pH: float) -> float:
    valid = [aa for aa in seq if aa in AA_TO_IDX]
    if not valid:
        return 0.0

    # N-terminus (positive)
    charge = 1.0 / (1.0 + 10 ** (pH - PKA_NTERM))

    # C-terminus (negative)
    charge -= 1.0 / (1.0 + 10 ** (PKA_CTERM - pH))

    counts = {aa: 0 for aa in PKA_SIDE}
    for aa in valid:
        if aa in counts:
            counts[aa] += 1

    # positive side chains
    for aa in ["K", "R", "H"]:
        if counts.get(aa, 0) > 0:
            pka = PKA_SIDE[aa]
            charge += counts[aa] * (1.0 / (1.0 + 10 ** (pH - pka)))

    # negative side chains
    for aa in ["D", "E", "C", "Y"]:
        if counts.get(aa, 0) > 0:
            pka = PKA_SIDE[aa]
            charge -= counts[aa] * (1.0 / (1.0 + 10 ** (pka - pH)))

    return float(charge)


def theoretical_pI(seq: str) -> float:
    valid = [aa for aa in seq if aa in AA_TO_IDX]
    if not valid:
        return 0.0

    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        charge = net_charge_at_pH(seq, mid)
        if charge > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def instability_index_approx(seq: str) -> float:
    valid = [aa for aa in seq if aa in AA_TO_IDX]
    if len(valid) < 2:
        return 0.0

    pairs = [valid[i] + valid[i + 1] for i in range(len(valid) - 1)]
    score = sum(INSTABILITY_PAIR_WEIGHT[p] for p in pairs)
    # escala semelhante a um índice
    return float((10.0 / len(valid)) * score)


def protparam_like_features(seq: str) -> np.ndarray:
    seq = clean_sequence(seq)
    return np.asarray([
        molecular_weight(seq),
        theoretical_pI(seq),
        gravy(seq),
        instability_index_approx(seq),
    ], dtype=np.float32)


def sequence_to_features(seq: str) -> np.ndarray:
    seq = clean_sequence(seq)

    aa_counts = np.zeros(AA_DIM, dtype=np.float32)
    dipep_counts = np.zeros(DIPEPTIDE_DIM, dtype=np.float32)
    group_counts = np.zeros(len(PHYSICOCHEM_GROUP_NAMES), dtype=np.float32)

    valid_idx = []
    valid_chars = []

    for aa in seq:
        idx = AA_TO_IDX.get(aa)
        if idx is not None:
            aa_counts[idx] += 1.0
            valid_idx.append(idx)
            valid_chars.append(aa)

    n_valid = len(valid_idx)

    if n_valid >= 2:
        prev = valid_idx[0]
        for curr in valid_idx[1:]:
            dipep_idx = prev * 20 + curr
            dipep_counts[dipep_idx] += 1.0
            prev = curr

    if n_valid > 0:
        aa_freq = aa_counts / n_valid if NORMALIZE_AA_FREQ else aa_counts.copy()
    else:
        aa_freq = aa_counts.copy()

    if n_valid > 1:
        dipep_freq = dipep_counts / (n_valid - 1) if NORMALIZE_DIPEPTIDE_FREQ else dipep_counts.copy()
    else:
        dipep_freq = dipep_counts.copy()

    seq_length = float(n_valid)
    log_length = float(math.log1p(n_valid)) if n_valid > 0 else 0.0
    entropy = shannon_entropy_from_freq(aa_freq) if n_valid > 0 else 0.0

    if n_valid > 0:
        for aa in valid_chars:
            for i, group_name in enumerate(PHYSICOCHEM_GROUP_NAMES):
                if aa in PHYSICOCHEM_GROUPS[group_name]:
                    group_counts[i] += 1.0
        group_props = group_counts / n_valid
    else:
        group_props = group_counts

    extra_features = []

    if USE_RAW_LENGTH:
        extra_features.append(seq_length)
    if USE_LOG_LENGTH:
        extra_features.append(log_length)

    extra_features.append(entropy)
    extra_features.extend(group_props.tolist())

    extra_features = np.asarray(extra_features, dtype=np.float32)
    seed_features = seed_summary_features(seq)
    protparam_features = protparam_like_features(seq)

    feat = np.empty(TOTAL_DIM, dtype=np.float32)
    feat[:AA_DIM] = aa_freq.astype(np.float32, copy=False)
    feat[AA_DIM:AA_DIM + DIPEPTIDE_DIM] = dipep_freq.astype(np.float32, copy=False)
    feat[AA_DIM + DIPEPTIDE_DIM:AA_DIM + DIPEPTIDE_DIM + EXTRA_DIM] = extra_features
    feat[AA_DIM + DIPEPTIDE_DIM + EXTRA_DIM:AA_DIM + DIPEPTIDE_DIM + EXTRA_DIM + SEED_STATS_DIM] = seed_features
    feat[AA_DIM + DIPEPTIDE_DIM + EXTRA_DIM + SEED_STATS_DIM:] = protparam_features

    return feat


def batch_sequences_to_features(seqs: list[str]) -> np.ndarray:
    X_chunk = np.empty((len(seqs), TOTAL_DIM), dtype=np.float32)
    for i, seq in enumerate(seqs):
        X_chunk[i] = sequence_to_features(seq)
    return X_chunk


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def main() -> None:
    start_time = time.time()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_CSV.resolve()}")

    ensure_output_dir(OUTPUT_DIR)

    print("[INFO] Contando linhas do CSV...")
    total_rows = count_csv_rows(INPUT_CSV)
    print(f"[INFO] Total de linhas de dados: {total_rows:,}")

    if total_rows == 0:
        raise ValueError("CSV sem linhas de dados.")

    processed_rows = prepare_run()
    save_feature_names()

    base_metadata = {
        "input_csv": str(INPUT_CSV.resolve()),
        "total_rows_csv": total_rows,
        "chunk_size": CHUNK_SIZE,
        "seq_col": SEQ_COL,
        "label_col": LABEL_COL,
        "header_col": HEADER_COL,
        "csv_separator": CSV_SEPARATOR,
        "amino_acids_order": AMINO_ACIDS,
        "aa_dim": AA_DIM,
        "dipeptide_dim": DIPEPTIDE_DIM,
        "extra_dim": EXTRA_DIM,
        "seed_stats_dim": SEED_STATS_DIM,
        "protparam_dim": PROTPARAM_DIM,
        "total_dim": TOTAL_DIM,
        "normalize_aa_freq": NORMALIZE_AA_FREQ,
        "normalize_dipeptide_freq": NORMALIZE_DIPEPTIDE_FREQ,
        "physicochem_groups": {k: "".join(sorted(v)) for k, v in PHYSICOCHEM_GROUPS.items()},
        "uses_raw_length": USE_RAW_LENGTH,
        "uses_log_length": USE_LOG_LENGTH,
        "seed_pattern": "2x1x2",
        "seed_window_size": WINDOW_SIZE,
        "seed_feature_names": SEED_FEATURE_NAMES,
        "protparam_feature_names": PROTPARAM_FEATURE_NAMES,
    }
    save_metadata(base_metadata)

    can_resume = processed_rows > 0 and X_DAT.exists() and Y_DAT.exists()

    if can_resume:
        X_mem, y_mem = load_existing_memmaps(total_rows)
        headers_mode = "a"
    else:
        X_mem, y_mem = prepare_output_memmaps(total_rows)
        headers_mode = "w"
        save_checkpoint(0)
        processed_rows = 0

    read_csv_kwargs = {
        "usecols": [SEQ_COL, LABEL_COL, HEADER_COL],
        "chunksize": CHUNK_SIZE,
        "dtype": {
            SEQ_COL: "string",
            LABEL_COL: "Int64",
            HEADER_COL: "string",
        },
        "sep": CSV_SEPARATOR,
        "encoding": CSV_ENCODING,
        "low_memory": True,
    }

    current_output_row = processed_rows
    skipped_before_resume = 0
    chunk_counter = 0
    valid_rows_written_this_run = 0

    print("[INFO] Iniciando processamento em chunks...")

    with HEADERS_TXT.open(headers_mode, encoding="utf-8", newline="\n") as f_headers:
        for chunk_df in pd.read_csv(INPUT_CSV, **read_csv_kwargs):
            chunk_counter += 1

            if skipped_before_resume < processed_rows:
                chunk_len = len(chunk_df)
                skipped_before_resume += chunk_len

                if skipped_before_resume <= processed_rows:
                    continue

                start_inside_chunk = processed_rows - (skipped_before_resume - chunk_len)
                chunk_df = chunk_df.iloc[start_inside_chunk:].copy()

            chunk_df = chunk_df.dropna(subset=[SEQ_COL, LABEL_COL])
            if chunk_df.empty:
                continue

            seqs = chunk_df[SEQ_COL].astype(str).tolist()
            labels = chunk_df[LABEL_COL].astype(int).tolist()
            headers = chunk_df[HEADER_COL].fillna("").astype(str).tolist()

            X_chunk = batch_sequences_to_features(seqs)
            n = len(seqs)

            row_slice = slice(current_output_row, current_output_row + n)
            X_mem[row_slice] = X_chunk
            y_mem[row_slice] = np.asarray(labels, dtype=np.int32)

            for header in headers:
                f_headers.write((header or "") + "\n")

            current_output_row += n
            valid_rows_written_this_run += n

            if valid_rows_written_this_run % REPORT_EVERY_ROWS == 0:
                elapsed = time.time() - start_time
                rate = valid_rows_written_this_run / elapsed if elapsed > 0 else 0
                print(
                    f"[INFO] {valid_rows_written_this_run:,} linhas gravadas nesta execução | "
                    f"posição global {current_output_row:,}/{total_rows:,} | "
                    f"{rate:,.2f} seq/s"
                )

            if chunk_counter % FLUSH_EVERY_CHUNKS == 0:
                X_mem.flush()
                y_mem.flush()
                f_headers.flush()
                save_checkpoint(current_output_row)
                print(f"[INFO] Checkpoint salvo em {current_output_row:,} linhas.")

    X_mem.flush()
    y_mem.flush()

    final_info = {
        "input_csv": str(INPUT_CSV.resolve()),
        "rows_in_input_csv": total_rows,
        "rows_written_valid": current_output_row,
        "shape_X": [current_output_row, TOTAL_DIM],
        "shape_y": [current_output_row],
        "dtype_X": str(np.dtype(X_DTYPE)),
        "dtype_y": str(np.dtype(Y_DTYPE)),
        "x_memmap_path": str(X_DAT.resolve()),
        "y_memmap_path": str(Y_DAT.resolve()),
        "headers_path": str(HEADERS_TXT.resolve()),
        "feature_names_path": str(FEATURE_NAMES_TXT.resolve()) if WRITE_FEATURE_NAMES_TXT else None,
        "chunk_size": CHUNK_SIZE,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    save_metadata(final_info)

    finalize_success()

    print("\n[SUCESSO] Processamento concluído.")
    print(f"Linhas válidas gravadas: {current_output_row:,}")
    print(f"Tempo total: {final_info['elapsed_seconds']:,} s")
    print(f"Total de features: {TOTAL_DIM:,}")
    print(f"Saída em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] Execução interrompida pelo usuário.")
        print("[INFO] O checkpoint foi mantido. Na próxima execução você poderá continuar.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRO] {type(e).__name__}: {e}")
        print("[INFO] O checkpoint e os arquivos parciais foram mantidos.")
        sys.exit(1)
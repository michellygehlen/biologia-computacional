from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURAÇÕES
# =========================================================
#origem para treinamento do primeiro modelo
#INPUT_CSV = Path("result_db/origem_600_features.csv")
#OUTPUT_DIR = Path("saida_features_600")

#teste para comparação com os outros modelos
INPUT_CSV = Path("result_db/resultado_debug_500k.csv")
OUTPUT_DIR = Path("saida_features_600_comparacao")

# Colunas do CSV
SEQ_COL = "Sequence"
LABEL_COL = "GroupID"
HEADER_COL = "FastaHeader"

# Alfabeto padrão de aminoácidos
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Dimensões
DIPEPTIDE_DIM = 400       # 20^2
FULL_DIM = 160000         # 400 * 400
PROJ_DIM = 600
WINDOW_SIZE = 5           # seed 2x1x2 => 2 + 1 + 2

# Processamento
DEFAULT_CHUNK_SIZE = 5000
FLUSH_EVERY_CHUNKS = 5
REPORT_EVERY_ROWS = 10000
PROJECTION_SEED = 42

# Reutiliza matriz de projeção se já existir
REUSE_PROJECTION = True


# =========================================================
# CAMINHOS DE SAÍDA
# =========================================================

X_DAT = OUTPUT_DIR / "X_features_600_float32.dat"
Y_DAT = OUTPUT_DIR / "y_labels_int32.dat"
HEADERS_TXT = OUTPUT_DIR / "headers.txt"
CHECKPOINT_JSON = OUTPUT_DIR / "checkpoint.json"
METADATA_JSON = OUTPUT_DIR / "metadata.json"
PROJECTION_NPY = OUTPUT_DIR / "projection_160000_to_600.npy"


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ask_chunk_size(default: int = DEFAULT_CHUNK_SIZE) -> int:
    while True:
        raw = input(f"Tamanho do chunk_size [padrão {default}]: ").strip()

        if raw == "":
            print(f"[INFO] Usando chunk_size padrão: {default}")
            return default

        try:
            value = int(raw)
            if value <= 0:
                print("Digite um número inteiro maior que zero.")
                continue
            print(f"[INFO] Usando chunk_size informado: {value}")
            return value
        except ValueError:
            print("Valor inválido. Digite um número inteiro, por exemplo 2000, 5000 ou 10000.")


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        total = sum(1 for _ in f) - 1
    return max(total, 0)


def dipeptide_index(a1: str, a2: str) -> int | None:
    idx1 = AA_TO_IDX.get(a1)
    idx2 = AA_TO_IDX.get(a2)
    if idx1 is None or idx2 is None:
        return None
    return idx1 * 20 + idx2


def sequence_seed_indices_2x1x2(seq: str) -> np.ndarray:
    seq = str(seq).strip().upper()
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


def build_or_load_projection_matrix(
    proj_path: Path,
    input_dim: int = FULL_DIM,
    output_dim: int = PROJ_DIM,
    seed: int = PROJECTION_SEED,
) -> np.ndarray:
    if REUSE_PROJECTION and proj_path.exists():
        print(f"[INFO] Carregando projeção existente: {proj_path}")
        return np.load(proj_path, mmap_mode="r")

    print("[INFO] Criando matriz de projeção aleatória...")
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((input_dim, output_dim), dtype=np.float32)

    norms = np.linalg.norm(proj, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    proj = proj / norms

    print(f"[INFO] Salvando projeção em: {proj_path}")
    np.save(proj_path, proj.astype(np.float32))

    del proj
    return np.load(proj_path, mmap_mode="r")


def sequence_to_600_features_from_projection(
    seq: str,
    proj_matrix: np.ndarray,
) -> np.ndarray:
    seed_indices = sequence_seed_indices_2x1x2(seq)

    if seed_indices.size == 0:
        return np.zeros(PROJ_DIM, dtype=np.float32)

    unique_idx, counts = np.unique(seed_indices, return_counts=True)
    feat = (proj_matrix[unique_idx].T @ counts.astype(np.float32)).astype(np.float32)
    return feat


def prepare_output_memmaps(total_rows: int) -> tuple[np.memmap, np.memmap]:
    print("[INFO] Criando memmaps de saída...")
    X_mem = np.memmap(
        X_DAT,
        dtype=np.float32,
        mode="w+",
        shape=(total_rows, PROJ_DIM),
    )
    y_mem = np.memmap(
        Y_DAT,
        dtype=np.int32,
        mode="w+",
        shape=(total_rows,),
    )
    return X_mem, y_mem


def load_existing_memmaps(total_rows: int) -> tuple[np.memmap, np.memmap]:
    X_mem = np.memmap(
        X_DAT,
        dtype=np.float32,
        mode="r+",
        shape=(total_rows, PROJ_DIM),
    )
    y_mem = np.memmap(
        Y_DAT,
        dtype=np.int32,
        mode="r+",
        shape=(total_rows,),
    )
    return X_mem, y_mem


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


def remove_file_if_exists(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def remove_output_files(keep_projection: bool = True) -> None:
    files_to_remove = [
        X_DAT,
        Y_DAT,
        HEADERS_TXT,
        CHECKPOINT_JSON,
        METADATA_JSON,
    ]

    if not keep_projection:
        files_to_remove.append(PROJECTION_NPY)

    for path in files_to_remove:
        remove_file_if_exists(path)


def checkpoint_exists() -> bool:
    return CHECKPOINT_JSON.exists()


def partial_output_exists() -> bool:
    return any(path.exists() for path in [X_DAT, Y_DAT, HEADERS_TXT, METADATA_JSON])


def ask_resume_or_restart() -> str:
    while True:
        print("\n[AVISO] Foi encontrado um checkpoint de execução anterior.")
        print("Digite:")
        print("  C = continuar de onde parou")
        print("  R = recomeçar do zero")
        choice = input("Sua escolha [C/R]: ").strip().upper()

        if choice in {"C", "R"}:
            return choice

        print("Opção inválida. Digite apenas C ou R.")


def prepare_run() -> int:
    ensure_output_dir(OUTPUT_DIR)

    has_checkpoint = checkpoint_exists()
    has_partial = partial_output_exists()

    if has_checkpoint:
        choice = ask_resume_or_restart()

        if choice == "C":
            processed_rows = load_checkpoint()
            print(f"[INFO] Continuando a partir de {processed_rows:,} linhas.")
            return processed_rows

        print("[INFO] Reiniciando do zero. Apagando saídas antigas...")
        remove_output_files(keep_projection=True)
        return 0

    if has_partial:
        print("[INFO] Saídas antigas encontradas sem checkpoint. Apagando para começar do zero...")
        remove_output_files(keep_projection=True)

    return 0


def finalize_success() -> None:
    remove_file_if_exists(CHECKPOINT_JSON)
    print("[INFO] Checkpoint removido após conclusão com sucesso.")


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def main() -> None:
    start_time = time.time()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_CSV.resolve()}")

    ensure_output_dir(OUTPUT_DIR)

    chunk_size = ask_chunk_size()

    print("[INFO] Contando linhas do CSV...")
    total_rows = count_csv_rows(INPUT_CSV)
    print(f"[INFO] Total de linhas de dados: {total_rows:,}")

    if total_rows == 0:
        raise ValueError("CSV sem linhas de dados.")

    processed_rows = prepare_run()

    base_metadata = {
        "input_csv": str(INPUT_CSV.resolve()),
        "total_rows_csv": total_rows,
        "chunk_size": chunk_size,
        "seq_col": SEQ_COL,
        "label_col": LABEL_COL,
        "header_col": HEADER_COL,
        "full_dim": FULL_DIM,
        "proj_dim": PROJ_DIM,
        "seed_pattern": "2x1x2",
        "projection_seed": PROJECTION_SEED,
    }
    save_metadata(base_metadata)

    proj_matrix = build_or_load_projection_matrix(PROJECTION_NPY)

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
        "chunksize": chunk_size,
        "dtype": {
            SEQ_COL: "string",
            LABEL_COL: "Int64",
            HEADER_COL: "string",
        },
        "encoding": "utf-8",
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

            if HEADER_COL in chunk_df.columns:
                headers = chunk_df[HEADER_COL].fillna("").astype(str).tolist()
            else:
                headers = [""] * len(chunk_df)

            for seq, label, header in zip(seqs, labels, headers):
                feat = sequence_to_600_features_from_projection(seq, proj_matrix)

                X_mem[current_output_row] = feat
                y_mem[current_output_row] = label
                f_headers.write((header or "") + "\n")

                current_output_row += 1
                valid_rows_written_this_run += 1

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
        "shape_X": [current_output_row, PROJ_DIM],
        "shape_y": [current_output_row],
        "dtype_X": "float32",
        "dtype_y": "int32",
        "x_memmap_path": str(X_DAT.resolve()),
        "y_memmap_path": str(Y_DAT.resolve()),
        "headers_path": str(HEADERS_TXT.resolve()),
        "projection_path": str(PROJECTION_NPY.resolve()),
        "chunk_size": chunk_size,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    save_metadata(final_info)

    finalize_success()

    print("\n[SUCESSO] Processamento concluído.")
    print(f"Linhas válidas gravadas: {current_output_row:,}")
    print(f"Tempo total: {final_info['elapsed_seconds']:,} s")
    print(f"Saída em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] Execução interrompida pelo usuário.")
        print("[INFO] O checkpoint foi mantido. Na próxima execução você poderá continuar ou reiniciar.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRO] {type(e).__name__}: {e}")
        print("[INFO] O checkpoint e os arquivos parciais foram mantidos.")
        print("[INFO] Na próxima execução o programa perguntará se deve continuar ou começar do zero.")
        sys.exit(1)
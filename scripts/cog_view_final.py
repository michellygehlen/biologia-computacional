import sqlite3
import pandas as pd
import os

# =========================================================
# CONFIG PADRÃO
# =========================================================

DB_PATH = "result_db/cog_relational.db"
DEFAULT_OUTPUT = "result_db/resultado_debug.csv"
DEFAULT_CHUNK = 50000

# =========================================================
# INPUT DO USUÁRIO
# =========================================================

print("\n=== EXPORTADOR DE CONSULTAS COG ===\n")

usar_padrao = input("Usar query padrão? [SELECT * FROM protein] (s/n): ").strip().lower()

if usar_padrao == "s":
    query = """
    SELECT *
    FROM protein
    """
else:
    print("\nDigite sua query SQL (finalize com ENTER duas vezes):\n")
    linhas = []
    while True:
        linha = input()
        if linha.strip() == "":
            break
        linhas.append(linha)
    query = "\n".join(linhas)

output_path = input(f"\nNome do arquivo de saída [{DEFAULT_OUTPUT}]: ").strip()
if output_path == "":
    output_path = DEFAULT_OUTPUT

chunk_input = input(f"Tamanho do chunk [{DEFAULT_CHUNK}]: ").strip()
chunk_size = int(chunk_input) if chunk_input else DEFAULT_CHUNK

# =========================================================
# EXECUÇÃO
# =========================================================

print("\nAbrindo banco...")
conn = sqlite3.connect(DB_PATH)

# remove arquivo antigo
if os.path.exists(output_path):
    os.remove(output_path)
    print("Arquivo antigo removido.")

print("\nExecutando consulta em chunks...\n")

chunk_iter = pd.read_sql_query(query, conn, chunksize=chunk_size)

total_linhas = 0

try:
    for i, chunk in enumerate(chunk_iter, start=1):
        modo = "w" if i == 1 else "a"
        header = True if i == 1 else False

        chunk.to_csv(
            output_path,
            mode=modo,
            header=header,
            index=False
        )

        total_linhas += len(chunk)

        print(f"Chunk {i} salvo | linhas: {len(chunk)} | total: {total_linhas}")

except Exception as e:
    print("\nERRO durante processamento:")
    print(e)

finally:
    conn.close()

print("\n===================================")
print(f"Arquivo final: {output_path}")
print(f"Total de linhas exportadas: {total_linhas}")
print("Concluído.\n")
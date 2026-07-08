import os
import re
import gc
import sqlite3
import pandas as pd


# =========================================================
# 0. CONFIGURAÇÃO
# =========================================================

RESULTS_DIR = "result_db"
DB_PATH = os.path.join(RESULTS_DIR, "cog_relational.db")
PROTEIN_PREVIEW_CSV = os.path.join(RESULTS_DIR, "protein_preview.csv")
PATHWAY_PREVIEW_CSV = os.path.join(RESULTS_DIR, "protein_pathway_preview.csv")
FUNCTION_PREVIEW_CSV = os.path.join(RESULTS_DIR, "protein_function_detail_preview.csv")

FILE_COGRED = "cog-24.cog.csv"
FILE_GENERED = "COGorg24.gene.tab"
FILE_FASTA = "COGorg24.faa"

FILE_MAPPING = "cog-24.mapping.tab"
FILE_DEFS = "cog-24.def.tab"
FILE_PATHWAYS = "cog-24.pathways.tab"
FILE_FUN = "cog-24.fun.tab"
FILE_FUN_GROUP = "cog-24.fun.group.tab"
FILE_ORG = "cog-24.org.csv"
FILE_TAX = "cog-24.tax.csv"


# =========================================================
# 1. UTILITÁRIOS
# =========================================================

def log_step(msg):
    print(msg, flush=True)


def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        log_step("Pasta 'results' criada.")
    else:
        log_step("Pasta 'results' encontrada.")


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw = f.read(4096)

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    for enc in ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "latin1"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                f.readline()
            return enc
        except Exception:
            pass

    raise ValueError(f"Não foi possível detectar o encoding de {file_path}")


def read_table(file_path, sep, header=None, usecols=None, comment=None):
    enc = detect_encoding(file_path)
    log_step(f"Lendo {file_path}...")

    try:
        df = pd.read_csv(
            file_path,
            sep=sep,
            header=header,
            encoding=enc,
            usecols=usecols,
            dtype="string",
            low_memory=True,
            engine="c"
        )
    except Exception:
        df = pd.read_csv(
            file_path,
            sep=sep,
            header=header,
            encoding=enc,
            usecols=usecols,
            dtype="string",
            low_memory=True,
            engine="python",
            on_bad_lines="skip",
            comment=comment
        )

    log_step(f"Arquivo carregado: {file_path} | linhas={len(df)} | colunas={len(df.columns)}")
    return df


def clean_object_columns(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].isin(["", "nan", "None", "NA", "<NA>"]), col] = pd.NA
    return df


def parse_genomic_range_series(series):
    extracted = series.astype("string").str.extract(r"(\d+)\.\.(\d+)")
    start_nt = pd.to_numeric(extracted[0], errors="coerce")
    end_nt = pd.to_numeric(extracted[1], errors="coerce")
    return start_nt, end_nt


def explode_function_letters(df, func_col="Function_combined"):
    """
    Cria tabela relacional ProteinWP <-> letra funcional
    """
    rows = []

    for rec in df[["ProteinWP", func_col]].dropna(subset=["ProteinWP"]).itertuples(index=False):
        protein_wp = rec[0]
        func_text = "" if pd.isna(rec[1]) else str(rec[1])

        for letter in func_text:
            if letter.isalpha():
                rows.append({
                    "ProteinWP": protein_wp,
                    "CategoryLetter": letter
                })

    out = pd.DataFrame(rows).drop_duplicates()
    return out


def parse_fasta_to_dataframe(file_path):
    enc = detect_encoding(file_path)
    log_step(f"Lendo FASTA {file_path}...")

    rows = []
    current_header = None
    seq_parts = []
    count = 0

    with open(file_path, "r", encoding=enc) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    rows.append(build_fasta_row(current_header, seq_parts))
                    count += 1
                    if count % 50000 == 0:
                        log_step(f"FASTA processado: {count} sequências")

                current_header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

        if current_header is not None:
            rows.append(build_fasta_row(current_header, seq_parts))
            count += 1

    fasta_df = pd.DataFrame(rows)
    fasta_df = clean_object_columns(fasta_df)

    log_step(f"FASTA carregado | sequências={len(fasta_df)}")
    return fasta_df


def build_fasta_row(header, seq_parts):
    sequence = "".join(seq_parts)
    parts = header.split()

    protein_wp = parts[0] if len(parts) >= 1 else pd.NA
    gene_hint = parts[1] if len(parts) >= 2 else pd.NA

    organism_match = re.search(r"\[(.*?)\]", header)
    organism_hint = organism_match.group(1).strip() if organism_match else pd.NA

    return {
        "ProteinWP": protein_wp,
        "FastaGeneHint": gene_hint,
        "FastaOrganismHint": organism_hint,
        "FastaHeader": header,
        "Sequence": sequence,
        "SequenceLengthAA": len(sequence)
    }


# =========================================================
# 2. CARREGAR TABELAS
# =========================================================

def load_all_tables():
    log_step("Carregando tabelas...")

    cog = read_table(FILE_COGRED, sep=",", header=None, usecols=list(range(13)))
    cog.columns = [
        "LocusTag",
        "Assembly",
        "ProteinWP",
        "ProteinLengthAA",
        "ProteinRangeAA",
        "ProteinLengthAA_repeat",
        "COG",
        "COG_best",
        "MembershipClass",
        "Bitscore",
        "Evalue",
        "COG_model_lengthAA",
        "COG_alignment_range"
    ]

    gene = read_table(FILE_GENERED, sep="\t", header=None, usecols=list(range(6)))
    gene.columns = [
        "LocusTag",
        "GenomicRange",
        "Strand",
        "Assembly",
        "Replicon",
        "ProteinWP"
    ]

    mapping = read_table(FILE_MAPPING, sep="\t", header=0)
    mapping = mapping.rename(columns={
        "COG no.": "COG",
        "Func": "Func_mapping",
        "Gene": "GeneSymbol_mapping",
        "COG annotation (cut to 42 characters)": "COG_annotation_short",
        "UniProt": "UniProt",
        "UniProt code": "UniProt_code",
        "Length,aa": "COG_length_mapping"
    })

    defs = read_table(FILE_DEFS, sep="\t", header=None, usecols=list(range(7)))
    defs.columns = [
        "COG",
        "Function",
        "Protein_name",
        "GeneSymbol_def",
        "Pathway_def",
        "PMID",
        "PDB"
    ]

    pathways = read_table(FILE_PATHWAYS, sep="\t", header=None, usecols=list(range(6)))
    pathways.columns = [
        "Pathway",
        "COG",
        "Function_path",
        "GeneSymbol_path",
        "Description",
        "EC"
    ]

    org = read_table(FILE_ORG, sep=",", header=None, usecols=list(range(4)))
    org.columns = [
        "Assembly",
        "Organism",
        "TaxID_org",
        "TaxGroup"
    ]

    tax = read_table(FILE_TAX, sep=",", header=None, usecols=list(range(3)))
    tax.columns = [
        "TaxGroup_tax",
        "Domain",
        "NCBI_TaxID_tax"
    ]

    fun = read_table(FILE_FUN, sep="\t", header=None, usecols=list(range(4)))
    fun.columns = [
        "CategoryLetter",
        "GroupID",
        "CategoryColor",
        "CategoryDescription"
    ]

    fun_group = read_table(FILE_FUN_GROUP, sep="\t", header=None, usecols=list(range(2)))
    fun_group.columns = [
        "GroupID",
        "FunctionalGroupName"
    ]

    fasta = parse_fasta_to_dataframe(FILE_FASTA)

    log_step("Limpando texto...")
    cog = clean_object_columns(cog)
    gene = clean_object_columns(gene)
    mapping = clean_object_columns(mapping)
    defs = clean_object_columns(defs)
    pathways = clean_object_columns(pathways)
    org = clean_object_columns(org)
    tax = clean_object_columns(tax)
    fun = clean_object_columns(fun)
    fun_group = clean_object_columns(fun_group)
    fasta = clean_object_columns(fasta)
    log_step("Limpeza concluída")

    log_step("Convertendo campos numéricos e coordenadas...")
    for col in ["ProteinLengthAA", "ProteinLengthAA_repeat", "Bitscore", "Evalue", "COG_model_lengthAA"]:
        if col in cog.columns:
            cog[col] = pd.to_numeric(cog[col], errors="coerce")

    if "COG_length_mapping" in mapping.columns:
        mapping["COG_length_mapping"] = pd.to_numeric(mapping["COG_length_mapping"], errors="coerce")

    gene["Start_nt"], gene["End_nt"] = parse_genomic_range_series(gene["GenomicRange"])
    gene["GeneLength_nt"] = gene["End_nt"] - gene["Start_nt"] + 1

    fasta["SequenceLengthAA"] = pd.to_numeric(fasta["SequenceLengthAA"], errors="coerce")
    log_step("Conversão concluída")

    log_step("Deduplicando tabelas centrais...")
    fasta = fasta.drop_duplicates(subset=["ProteinWP"])
    gene = gene.drop_duplicates(subset=["ProteinWP"])
    cog = cog.drop_duplicates(subset=["ProteinWP"])
    mapping = mapping.drop_duplicates(subset=["COG"])
    defs = defs.drop_duplicates(subset=["COG"])
    org = org.drop_duplicates(subset=["Assembly"])
    tax = tax.drop_duplicates(subset=["TaxGroup_tax"])
    log_step("Deduplicação concluída")

    log_step("Integrando categorias funcionais...")
    fun_full = fun.merge(fun_group, on="GroupID", how="left")
    fun_full = fun_full.drop_duplicates(subset=["CategoryLetter"])
    log_step(f"Categorias funcionais carregadas | letras={len(fun_full)}")

    return {
        "fasta": fasta,
        "gene": gene,
        "cog": cog,
        "defs": defs,
        "mapping": mapping,
        "pathways_raw": pathways,
        "org": org,
        "tax": tax,
        "fun_full": fun_full
    }


# =========================================================
# 3. TABELA PRINCIPAL PROTEIN
# =========================================================

def build_protein_table(tables):
    log_step("Montando tabela principal 'protein'...")

    fasta = tables["fasta"].copy()
    gene = tables["gene"]
    cog = tables["cog"]
    defs = tables["defs"]
    mapping = tables["mapping"]
    org = tables["org"]
    tax = tables["tax"]

    # FASTA campo 1 -> gene campo 6
    log_step("Merge protein ↔ gene...")
    df = fasta.merge(gene, how="left", on="ProteinWP", suffixes=("", "_gene"))
    log_step(f"Após merge com gene | linhas={len(df)} | colunas={len(df.columns)}")

    # FASTA campo 1 -> cog campo 3
    log_step("Merge protein ↔ cog...")
    df = df.merge(cog, how="left", on="ProteinWP", suffixes=("", "_cog"))
    log_step(f"Após merge com cog | linhas={len(df)} | colunas={len(df.columns)}")

    # COG -> defs
    log_step("Merge com defs...")
    df = df.merge(defs, how="left", on="COG")
    log_step(f"Após merge com defs | linhas={len(df)} | colunas={len(df.columns)}")

    # COG -> mapping
    log_step("Merge com mapping...")
    df = df.merge(mapping, how="left", on="COG")
    log_step(f"Após merge com mapping | linhas={len(df)} | colunas={len(df.columns)}")

    # Assembly -> org
    log_step("Merge com org...")
    df = df.merge(org, how="left", on="Assembly")
    log_step(f"Após merge com org | linhas={len(df)} | colunas={len(df.columns)}")

    # TaxGroup -> tax
    log_step("Merge com tax...")
    df = df.merge(tax, how="left", left_on="TaxGroup", right_on="TaxGroup_tax")
    log_step(f"Após merge com tax | linhas={len(df)} | colunas={len(df.columns)}")

    # Function_combined
    log_step("Criando Function_combined...")
    df["Function_combined"] = df["Function"]
    missing_func = df["Function_combined"].isna() | (df["Function_combined"].astype("string").str.strip() == "")
    df.loc[missing_func, "Function_combined"] = df.loc[missing_func, "Func_mapping"]
    df["Function_combined"] = df["Function_combined"].fillna("").astype("string")

    # consistência de tamanho
    if "ProteinLengthAA" in df.columns:
        df["SequenceLengthMatchesProteinLength"] = (
            df["SequenceLengthAA"].notna()
            & df["ProteinLengthAA"].notna()
            & (df["SequenceLengthAA"] == df["ProteinLengthAA"])
        ).astype("int8")
    else:
        df["SequenceLengthMatchesProteinLength"] = 0

    return df


# =========================================================
# 4. TABELA RELACIONAL PROTEIN_PATHWAY
# =========================================================

def build_protein_pathway_table(protein_df, pathways_raw):
    log_step("Montando tabela relacional 'protein_pathway'...")

    base = protein_df[["ProteinWP", "COG"]].dropna(subset=["ProteinWP"]).drop_duplicates()

    # COG -> pathways_raw campo 2
    protein_pathway = base.merge(
        pathways_raw,
        how="left",
        on="COG"
    )

    protein_pathway = protein_pathway[[
        "ProteinWP", "COG", "Pathway", "Description", "EC", "Function_path", "GeneSymbol_path"
    ]].dropna(subset=["ProteinWP"])

    protein_pathway = protein_pathway.drop_duplicates()

    log_step(f"Tabela protein_pathway pronta | linhas={len(protein_pathway)}")
    return protein_pathway


# =========================================================
# 5. TABELA RELACIONAL PROTEIN_FUNCTION_LETTER
# =========================================================

def build_protein_function_tables(protein_df, fun_full):
    log_step("Montando tabelas relacionais de função...")

    protein_letter = explode_function_letters(protein_df, func_col="Function_combined")

    # detalhe funcional
    protein_detail = protein_letter.merge(
        fun_full,
        how="left",
        on="CategoryLetter"
    )

    protein_letter = protein_letter.drop_duplicates()
    protein_detail = protein_detail.drop_duplicates()

    log_step(f"Tabela protein_function_letter pronta | linhas={len(protein_letter)}")
    log_step(f"Tabela protein_function_detail pronta | linhas={len(protein_detail)}")

    return protein_letter, protein_detail


# =========================================================
# 6. SALVAR SQLITE
# =========================================================

def save_to_sqlite(tables, protein_df, protein_pathway, protein_letter, protein_detail):
    log_step("Criando banco SQLite...")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    log_step("Salvando tabelas brutas...")
    tables["fasta"].to_sql("fasta", conn, index=False)
    tables["gene"].to_sql("gene", conn, index=False)
    tables["cog"].to_sql("cog", conn, index=False)
    tables["defs"].to_sql("defs", conn, index=False)
    tables["mapping"].to_sql("mapping", conn, index=False)
    tables["pathways_raw"].to_sql("pathways_raw", conn, index=False)
    tables["org"].to_sql("org", conn, index=False)
    tables["tax"].to_sql("tax", conn, index=False)
    tables["fun_full"].to_sql("fun_full", conn, index=False)

    log_step("Salvando tabelas relacionais...")
    protein_df.to_sql("protein", conn, index=False)
    protein_pathway.to_sql("protein_pathway", conn, index=False)
    protein_letter.to_sql("protein_function_letter", conn, index=False)
    protein_detail.to_sql("protein_function_detail", conn, index=False)

    cur = conn.cursor()

    log_step("Criando índices...")
    cur.execute("CREATE INDEX idx_fasta_proteinwp ON fasta(ProteinWP)")
    cur.execute("CREATE INDEX idx_gene_proteinwp ON gene(ProteinWP)")
    cur.execute("CREATE INDEX idx_cog_proteinwp ON cog(ProteinWP)")
    cur.execute("CREATE INDEX idx_cog_cog ON cog(COG)")
    cur.execute("CREATE INDEX idx_defs_cog ON defs(COG)")
    cur.execute("CREATE INDEX idx_mapping_cog ON mapping(COG)")
    cur.execute("CREATE INDEX idx_pathways_raw_cog ON pathways_raw(COG)")
    cur.execute("CREATE INDEX idx_org_assembly ON org(Assembly)")
    cur.execute("CREATE INDEX idx_tax_taxgroup ON tax(TaxGroup_tax)")

    cur.execute("CREATE INDEX idx_protein_proteinwp ON protein(ProteinWP)")
    cur.execute("CREATE INDEX idx_protein_cog ON protein(COG)")
    cur.execute("CREATE INDEX idx_pp_proteinwp ON protein_pathway(ProteinWP)")
    cur.execute("CREATE INDEX idx_pp_cog ON protein_pathway(COG)")
    cur.execute("CREATE INDEX idx_pfl_proteinwp ON protein_function_letter(ProteinWP)")
    cur.execute("CREATE INDEX idx_pfd_proteinwp ON protein_function_detail(ProteinWP)")
    cur.execute("CREATE INDEX idx_pfd_letter ON protein_function_detail(CategoryLetter)")

    conn.commit()
    conn.close()

    log_step(f"Banco SQLite salvo em: {DB_PATH}")


# =========================================================
# 7. PREVIEW EM CSV
# =========================================================

def save_previews(protein_df, protein_pathway, protein_detail):
    log_step("Salvando previews em CSV...")
    protein_df.head(2000).to_csv(PROTEIN_PREVIEW_CSV, index=False)
    protein_pathway.head(2000).to_csv(PATHWAY_PREVIEW_CSV, index=False)
    protein_detail.head(2000).to_csv(FUNCTION_PREVIEW_CSV, index=False)
    log_step("Previews salvos.")


# =========================================================
# 8. CONSULTAS EXEMPLO
# =========================================================

def show_example_queries(conn, protein_wp):
    print("\n================ EXEMPLO DE CONSULTA ================\n")

    print("TABELA protein\n")
    df1 = pd.read_sql_query(
        "SELECT * FROM protein WHERE ProteinWP = ?",
        conn,
        params=[protein_wp]
    )
    print(df1.T)

    print("\nTABELA protein_pathway\n")
    df2 = pd.read_sql_query(
        "SELECT ProteinWP, Pathway, Description, EC FROM protein_pathway WHERE ProteinWP = ?",
        conn,
        params=[protein_wp]
    )
    print(df2)

    print("\nTABELA protein_function_detail\n")
    df3 = pd.read_sql_query(
        """
        SELECT ProteinWP, CategoryLetter, CategoryDescription, FunctionalGroupName
        FROM protein_function_detail
        WHERE ProteinWP = ?
        """,
        conn,
        params=[protein_wp]
    )
    print(df3)


# =========================================================
# 9. MAIN
# =========================================================

def main():
    ensure_results_dir()

    tables = load_all_tables()

    protein_df = build_protein_table(tables)
    protein_pathway = build_protein_pathway_table(protein_df, tables["pathways_raw"])
    protein_letter, protein_detail = build_protein_function_tables(protein_df, tables["fun_full"])

    save_previews(protein_df, protein_pathway, protein_detail)
    save_to_sqlite(tables, protein_df, protein_pathway, protein_letter, protein_detail)

    print("\n================ RESUMO ================")
    print(f"Proteínas no FASTA: {len(tables['fasta'])}")
    print(f"Tabela protein: {len(protein_df)} linhas")
    print(f"Tabela protein_pathway: {len(protein_pathway)} linhas")
    print(f"Tabela protein_function_letter: {len(protein_letter)} linhas")
    print(f"Tabela protein_function_detail: {len(protein_detail)} linhas")
    print(f"Banco criado: {DB_PATH}")
    print(f"Preview protein: {PROTEIN_PREVIEW_CSV}")
    print(f"Preview pathway: {PATHWAY_PREVIEW_CSV}")
    print(f"Preview function: {FUNCTION_PREVIEW_CSV}")

    conn = sqlite3.connect(DB_PATH)

    example_series = protein_df["ProteinWP"].dropna().astype(str)
    if len(example_series) > 0:
        example_wp = example_series.iloc[0]
        print(f"\nExibindo exemplo para ProteinWP = {example_wp}")
        show_example_queries(conn, example_wp)

    conn.close()

    del tables, protein_df, protein_pathway, protein_letter, protein_detail
    gc.collect()

    print("\nPipeline concluído com sucesso.")


if __name__ == "__main__":
    main()
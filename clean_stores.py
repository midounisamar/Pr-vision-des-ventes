"""
Détection et nettoyage de stores.csv avec SQL (DuckDB).

Usage:
    pip install duckdb
    py -3 clean_stores.py
"""

from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).parent
NETTOYAGE_SQL = BASE_DIR / "sql3" / "22_nettoyage.sql"
OUTPUT_CSV = BASE_DIR / "stores_clean.csv"


def strip_sql_comments(sql: str) -> str:
    return "\n".join(
        line for line in sql.splitlines()
        if not line.strip().startswith("--")
    )


def run_sql_file(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    sql = strip_sql_comments(path.read_text(encoding="utf-8"))
    for statement in sql.split(";"):
        query = statement.strip()
        if query:
            con.execute(query)


def run_detection(con: duckdb.DuckDBPyConnection) -> None:
    print("=" * 60)
    print("DETECTION - Valeurs manquantes, doublons, incohérences, outliers")
    print("=" * 60)

    con.execute(
        "CREATE OR REPLACE VIEW stores AS "
        "SELECT * FROM read_csv_auto('stores.csv', header=true)"
    )

    checks = {
        "Valeurs manquantes": """
            SELECT
                SUM(CASE WHEN store_nbr IS NULL THEN 1 ELSE 0 END) AS store_nbr,
                SUM(CASE WHEN city IS NULL THEN 1 ELSE 0 END)      AS city,
                SUM(CASE WHEN state IS NULL THEN 1 ELSE 0 END)     AS state,
                SUM(CASE WHEN type IS NULL THEN 1 ELSE 0 END)      AS type,
                SUM(CASE WHEN cluster IS NULL THEN 1 ELSE 0 END)   AS cluster
            FROM stores
        """,
        "Doublons (store_nbr)": """
            SELECT COUNT(*) - COUNT(DISTINCT store_nbr) AS nb_doublons
            FROM stores
        """,
        "Doublons (clé métier complète)": """
            SELECT
                COUNT(*) - COUNT(DISTINCT (store_nbr, city, state, type, cluster))
                AS nb_doublons
            FROM stores
        """,
        "Incohérences": """
            SELECT
                SUM(CASE WHEN TRY_CAST(store_nbr AS INTEGER) IS NULL THEN 1 ELSE 0 END)
                    AS store_nbr_invalide,
                SUM(CASE WHEN store_nbr <= 0 OR store_nbr > 54 THEN 1 ELSE 0 END)
                    AS store_nbr_hors_plage,
                SUM(CASE WHEN city IS NULL OR TRIM(city) = '' THEN 1 ELSE 0 END)
                    AS city_vide,
                SUM(CASE WHEN type NOT IN ('A', 'B', 'C', 'D', 'E') THEN 1 ELSE 0 END)
                    AS type_inconnu,
                SUM(CASE WHEN cluster < 1 OR cluster > 17 THEN 1 ELSE 0 END)
                    AS cluster_hors_plage
            FROM stores
        """,
        "store_nbr multi-villes": """
            SELECT COUNT(*) AS nb_store_nbr_incoherents
            FROM (
                SELECT store_nbr
                FROM stores
                GROUP BY store_nbr
                HAVING COUNT(DISTINCT city) > 1
            ) t
        """,
        "Clusters rares (< 2 magasins)": """
            SELECT COUNT(*) AS nb_clusters_rares
            FROM (
                SELECT cluster
                FROM stores
                GROUP BY cluster
                HAVING COUNT(*) < 2
            ) t
        """,
    }

    for title, query in checks.items():
        print(f"\n--- {title} ---")
        print(con.execute(query).fetchdf().to_string(index=False))


def run_cleaning(con: duckdb.DuckDBPyConnection) -> None:
    print("\n" + "=" * 60)
    print("NETTOYAGE - Export vers stores_clean.csv")
    print("=" * 60)

    run_sql_file(con, NETTOYAGE_SQL)

    stats = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM read_csv_auto('stores.csv', header=true)) AS lignes_avant,
            (SELECT COUNT(*) FROM stores_step1) AS apres_normalisation,
            (SELECT COUNT(*) FROM stores_step2) AS apres_dedoublonnage,
            COUNT(*) AS lignes_apres
        FROM stores_clean
    """).fetchone()

    print(f"Lignes avant           : {stats[0]:,}")
    print(f"Après normalisation    : {stats[1]:,}")
    print(f"Après dédoublonnage    : {stats[2]:,}")
    print(f"Lignes finales         : {stats[3]:,}")
    print(f"Lignes supprimées      : {stats[0] - stats[3]:,}")
    print(f"Fichier créé           : {OUTPUT_CSV}")


def main() -> None:
    con = duckdb.connect()
    run_detection(con)
    run_cleaning(con)
    con.close()


if __name__ == "__main__":
    main()

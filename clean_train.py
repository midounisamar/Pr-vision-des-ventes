"""
Détection et nettoyage de train.csv avec SQL (DuckDB).

Usage:
    pip install duckdb
    py -3 clean_train.py
"""

from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).parent
DETECTION_SQL = BASE_DIR / "sql" / "01_detection.sql"
NETTOYAGE_SQL = BASE_DIR / "sql" / "02_nettoyage.sql"
OUTPUT_CSV = BASE_DIR / "train_clean.csv"


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
        "CREATE OR REPLACE VIEW train AS "
        "SELECT * FROM read_csv_auto('train.csv', header=true)"
    )

    checks = {
        "Valeurs manquantes": """
            SELECT
                SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END)          AS id,
                SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END)        AS date,
                SUM(CASE WHEN store_nbr IS NULL THEN 1 ELSE 0 END)    AS store_nbr,
                SUM(CASE WHEN family IS NULL THEN 1 ELSE 0 END)      AS family,
                SUM(CASE WHEN sales IS NULL THEN 1 ELSE 0 END)       AS sales,
                SUM(CASE WHEN onpromotion IS NULL THEN 1 ELSE 0 END) AS onpromotion
            FROM train
        """,
        "Doublons (id)": """
            SELECT COUNT(*) - COUNT(DISTINCT id) AS nb_doublons FROM train
        """,
        "Doublons (date + store + family)": """
            SELECT COUNT(*) - COUNT(DISTINCT (date, store_nbr, family)) AS nb_doublons
            FROM train
        """,
        "Incohérences": """
            SELECT
                SUM(CASE WHEN sales < 0 THEN 1 ELSE 0 END)                         AS ventes_negatives,
                SUM(CASE WHEN onpromotion < 0 THEN 1 ELSE 0 END)                   AS promo_negatives,
                SUM(CASE WHEN store_nbr <= 0 OR store_nbr > 54 THEN 1 ELSE 0 END)  AS magasin_hors_plage,
                SUM(CASE WHEN TRY_CAST(date AS DATE) IS NULL THEN 1 ELSE 0 END)    AS date_invalide
            FROM train
        """,
        "Outliers IQR (sales > 0)": """
            WITH stats AS (
                SELECT
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY sales) AS q1,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY sales) AS q3
                FROM train WHERE sales > 0
            )
            SELECT COUNT(*) AS nb_outliers
            FROM train t, stats s
            WHERE t.sales > s.q3 + 1.5 * (s.q3 - s.q1)
        """,
        "Outliers > P99": """
            WITH p99 AS (
                SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY sales) AS seuil FROM train
            )
            SELECT COUNT(*) AS nb_outliers, MAX(t.sales) AS vente_max
            FROM train t, p99
            WHERE t.sales > p99.seuil
        """,
    }

    for title, query in checks.items():
        print(f"\n--- {title} ---")
        print(con.execute(query).fetchdf().to_string(index=False))


def run_cleaning(con: duckdb.DuckDBPyConnection) -> None:
    print("\n" + "=" * 60)
    print("NETTOYAGE - Export vers train_clean.csv")
    print("=" * 60)

    run_sql_file(con, NETTOYAGE_SQL)

    stats = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM read_csv_auto('train.csv', header=true)) AS lignes_avant,
            COUNT(*) AS lignes_apres,
            SUM(CASE WHEN sales_winsorized THEN 1 ELSE 0 END) AS ventes_plafonnees
        FROM train_clean
    """).fetchone()

    print(f"Lignes avant  : {stats[0]:,}")
    print(f"Lignes après  : {stats[1]:,}")
    print(f"Ventes plafonnées (P99) : {stats[2]:,}")
    print(f"Fichier créé  : {OUTPUT_CSV}")


def main() -> None:
    con = duckdb.connect()
    run_detection(con)
    run_cleaning(con)
    con.close()


if __name__ == "__main__":
    main()

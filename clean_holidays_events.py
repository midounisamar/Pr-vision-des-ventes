"""
Détection et nettoyage de holidays_events.csv avec SQL (DuckDB).

Usage:
    pip install duckdb
    py -3 clean_holidays_events.py
"""

from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).parent
NETTOYAGE_SQL = BASE_DIR / "sql2" / "12_nettoyage.sql"
OUTPUT_CSV = BASE_DIR / "holidays_events_clean.csv"


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
        "CREATE OR REPLACE VIEW holidays_events AS "
        "SELECT * FROM read_csv_auto('holidays_events.csv', header=true)"
    )

    checks = {
        "Valeurs manquantes": """
            SELECT
                SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END)           AS date,
                SUM(CASE WHEN type IS NULL THEN 1 ELSE 0 END)           AS type,
                SUM(CASE WHEN locale IS NULL THEN 1 ELSE 0 END)         AS locale,
                SUM(CASE WHEN locale_name IS NULL THEN 1 ELSE 0 END)    AS locale_name,
                SUM(CASE WHEN description IS NULL THEN 1 ELSE 0 END)    AS description,
                SUM(CASE WHEN transferred IS NULL THEN 1 ELSE 0 END)    AS transferred
            FROM holidays_events
        """,
        "Doublons (clé événement)": """
            SELECT
                COUNT(*) - COUNT(DISTINCT (date, type, locale, locale_name, description, transferred))
                AS nb_doublons
            FROM holidays_events
        """,
        "Incohérences": """
            SELECT
                SUM(CASE WHEN TRY_CAST(date AS DATE) IS NULL THEN 1 ELSE 0 END) AS date_invalide,
                SUM(CASE WHEN type IS NULL OR TRIM(type) = '' THEN 1 ELSE 0 END) AS type_vide,
                SUM(CASE WHEN locale IS NULL OR TRIM(locale) = '' THEN 1 ELSE 0 END) AS locale_vide,
                SUM(CASE WHEN type NOT IN (
                    'Holiday', 'Transfer', 'Additional', 'Bridge', 'Work Day', 'Event'
                ) THEN 1 ELSE 0 END) AS type_inconnu,
                SUM(CASE WHEN locale NOT IN ('Local', 'Regional', 'National') THEN 1 ELSE 0 END)
                    AS locale_inconnu
            FROM holidays_events
        """,
        "Dates hors plage (2012-2017)": """
            SELECT COUNT(*) AS nb_hors_plage
            FROM holidays_events
            WHERE TRY_CAST(date AS DATE) < DATE '2012-01-01'
               OR TRY_CAST(date AS DATE) > DATE '2017-12-31'
        """,
        "Dates avec charge anormale": """
            WITH events_par_date AS (
                SELECT date, COUNT(*) AS nb_events
                FROM holidays_events
                GROUP BY date
            ),
            seuil AS (
                SELECT AVG(nb_events) + 2 * STDDEV(nb_events) AS limite
                FROM events_par_date
            )
            SELECT COUNT(*) AS nb_dates_anormales
            FROM events_par_date e
            CROSS JOIN seuil s
            WHERE e.nb_events > s.limite
        """,
    }

    for title, query in checks.items():
        print(f"\n--- {title} ---")
        print(con.execute(query).fetchdf().to_string(index=False))


def run_cleaning(con: duckdb.DuckDBPyConnection) -> None:
    print("\n" + "=" * 60)
    print("NETTOYAGE - Export vers holidays_events_clean.csv")
    print("=" * 60)

    run_sql_file(con, NETTOYAGE_SQL)

    stats = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM read_csv_auto('holidays_events.csv', header=true)) AS lignes_avant,
            (SELECT COUNT(*) FROM holidays_events_step1) AS apres_normalisation,
            (SELECT COUNT(*) FROM holidays_events_step2) AS apres_dedoublonnage,
            COUNT(*) AS lignes_apres
        FROM holidays_events_clean
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

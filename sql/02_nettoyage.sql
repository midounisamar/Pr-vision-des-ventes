-- ============================================================
-- 02 - NETTOYAGE des données train.csv
-- Moteur : DuckDB
-- Sortie : train_clean.csv
-- ============================================================

CREATE OR REPLACE VIEW train_raw AS
SELECT * FROM read_csv_auto('train.csv', header = true);


-- ------------------------------------------------------------
-- Étape 1 : normalisation et correction des incohérences
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW train_step1 AS
SELECT
    id,
    CAST(date AS DATE)                              AS date,
    store_nbr,
    TRIM(family)                                    AS family,
    CASE WHEN sales < 0 THEN 0 ELSE sales END       AS sales,
    CASE
        WHEN onpromotion < 0 THEN 0
        ELSE onpromotion
    END                                             AS onpromotion
FROM train_raw
WHERE id IS NOT NULL
  AND date IS NOT NULL
  AND store_nbr IS NOT NULL
  AND family IS NOT NULL
  AND sales IS NOT NULL
  AND onpromotion IS NOT NULL
  AND TRIM(family) <> ''
  AND store_nbr BETWEEN 1 AND 54
  AND TRY_CAST(date AS DATE) IS NOT NULL;


-- ------------------------------------------------------------
-- Étape 2 : suppression des doublons (garder la 1re occurrence)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW train_step2 AS
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY date, store_nbr, family
            ORDER BY id
        ) AS rn
    FROM train_step1
) dedup
WHERE rn = 1;


-- ------------------------------------------------------------
-- Étape 3 : traitement des valeurs aberrantes (winsorisation P99)
-- Les ventes > 99e percentile sont plafonnées au seuil P99
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW train_clean AS
WITH seuil AS (
    SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY sales) AS p99
    FROM train_step2
)
SELECT
    t.id,
    t.date,
    t.store_nbr,
    t.family,
    CASE
        WHEN t.sales > s.p99 THEN s.p99
        ELSE t.sales
    END AS sales,
    t.onpromotion,
    CASE WHEN t.sales > s.p99 THEN TRUE ELSE FALSE END AS sales_winsorized
FROM train_step2 t
CROSS JOIN seuil s;


-- ------------------------------------------------------------
-- Export du fichier nettoyé (sans la colonne technique)
-- ------------------------------------------------------------
COPY (
    SELECT id, date, store_nbr, family, sales, onpromotion
    FROM train_clean
    ORDER BY id
) TO 'train_clean.csv' (HEADER, DELIMITER ',');

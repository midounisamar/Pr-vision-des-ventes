-- ============================================================
-- 02 - NETTOYAGE des données stores.csv
-- Moteur : DuckDB
-- Sortie : stores_clean.csv
-- ============================================================

CREATE OR REPLACE VIEW stores_raw AS
SELECT * FROM read_csv_auto('stores.csv', header = true);


-- ------------------------------------------------------------
-- Étape 1 : normalisation et correction des incohérences
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW stores_step1 AS
SELECT
    CAST(store_nbr AS INTEGER) AS store_nbr,
    TRIM(city)                 AS city,
    TRIM(state)                AS state,
    TRIM(type)                 AS type,
    CAST(cluster AS INTEGER)   AS cluster
FROM stores_raw
WHERE store_nbr IS NOT NULL
  AND city IS NOT NULL
  AND state IS NOT NULL
  AND type IS NOT NULL
  AND cluster IS NOT NULL
  AND TRIM(city) <> ''
  AND TRIM(state) <> ''
  AND TRIM(type) <> ''
  AND TRY_CAST(store_nbr AS INTEGER) IS NOT NULL
  AND TRY_CAST(cluster AS INTEGER) IS NOT NULL
  AND type IN ('A', 'B', 'C', 'D', 'E')
  AND cluster BETWEEN 1 AND 17;


-- ------------------------------------------------------------
-- Étape 2 : suppression des doublons (garder la 1re occurrence par store_nbr)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW stores_step2 AS
SELECT
    store_nbr,
    city,
    state,
    type,
    cluster
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY store_nbr
            ORDER BY store_nbr
        ) AS rn
    FROM stores_step1
) dedup
WHERE rn = 1;


-- ------------------------------------------------------------
-- Étape 3 : filtrage des valeurs aberrantes
-- (store_nbr hors plage du projet : 1 à 54)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW stores_clean AS
SELECT
    store_nbr,
    city,
    state,
    type,
    cluster
FROM stores_step2
WHERE store_nbr BETWEEN 1 AND 54;


-- ------------------------------------------------------------
-- Export du fichier nettoyé
-- ------------------------------------------------------------
COPY (
    SELECT store_nbr, city, state, type, cluster
    FROM stores_clean
    ORDER BY store_nbr
) TO 'stores_clean.csv' (HEADER, DELIMITER ',');

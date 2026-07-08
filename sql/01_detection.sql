-- ============================================================
-- 01 - DETECTION : valeurs manquantes, doublons,
--                  incohérences, valeurs aberrantes
-- Dataset : train.csv (Store Sales - Kaggle)
-- Moteur  : DuckDB (compatible read_csv_auto)
-- ============================================================

CREATE OR REPLACE VIEW train AS
SELECT * FROM read_csv_auto('train.csv', header = true);


-- ------------------------------------------------------------
-- 1. VALEURS MANQUANTES (NULL)
-- ------------------------------------------------------------
SELECT
    'valeurs_manquantes' AS controle,
    COUNT(*)             AS total_lignes,
    SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END)           AS id_null,
    SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END)         AS date_null,
    SUM(CASE WHEN store_nbr IS NULL THEN 1 ELSE 0 END)    AS store_nbr_null,
    SUM(CASE WHEN family IS NULL THEN 1 ELSE 0 END)       AS family_null,
    SUM(CASE WHEN sales IS NULL THEN 1 ELSE 0 END)        AS sales_null,
    SUM(CASE WHEN onpromotion IS NULL THEN 1 ELSE 0 END)  AS onpromotion_null
FROM train;


-- Détail par colonne (format long)
SELECT
    col AS colonne,
    nb_nulls,
    ROUND(100.0 * nb_nulls / total, 4) AS pct_null
FROM (
    SELECT COUNT(*) AS total FROM train
) t
CROSS JOIN (
    SELECT 'id'           AS col, SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END)          AS nb_nulls FROM train UNION ALL
    SELECT 'date',                  SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END)                     FROM train UNION ALL
    SELECT 'store_nbr',             SUM(CASE WHEN store_nbr IS NULL THEN 1 ELSE 0 END)                FROM train UNION ALL
    SELECT 'family',                SUM(CASE WHEN family IS NULL THEN 1 ELSE 0 END)                   FROM train UNION ALL
    SELECT 'sales',                 SUM(CASE WHEN sales IS NULL THEN 1 ELSE 0 END)                    FROM train UNION ALL
    SELECT 'onpromotion',           SUM(CASE WHEN onpromotion IS NULL THEN 1 ELSE 0 END)              FROM train
) v;


-- ------------------------------------------------------------
-- 2. DOUBLONS
-- ------------------------------------------------------------

-- Doublons sur la clé technique (id)
SELECT
    'doublons_id' AS controle,
    COUNT(*) - COUNT(DISTINCT id) AS nb_doublons_id
FROM train;

-- Doublons sur la clé métier (date + magasin + famille)
SELECT
    'doublons_cle_metier' AS controle,
    COUNT(*) - COUNT(DISTINCT (date, store_nbr, family)) AS nb_doublons_metier
FROM train;

-- Exemples de lignes dupliquées (clé métier)
SELECT
    date,
    store_nbr,
    family,
    COUNT(*) AS occurrences
FROM train
GROUP BY date, store_nbr, family
HAVING COUNT(*) > 1
ORDER BY occurrences DESC
LIMIT 20;


-- ------------------------------------------------------------
-- 3. INCOHÉRENCES
-- ------------------------------------------------------------
SELECT
    'incoherences' AS controle,
    SUM(CASE WHEN sales < 0 THEN 1 ELSE 0 END)                              AS ventes_negatives,
    SUM(CASE WHEN onpromotion < 0 THEN 1 ELSE 0 END)                        AS promo_negatives,
    SUM(CASE WHEN store_nbr <= 0 THEN 1 ELSE 0 END)                         AS magasin_invalide,
    SUM(CASE WHEN family IS NULL OR TRIM(family) = '' THEN 1 ELSE 0 END)      AS famille_vide,
    SUM(CASE WHEN TRY_CAST(date AS DATE) IS NULL THEN 1 ELSE 0 END)           AS date_invalide,
    SUM(CASE WHEN onpromotion > 0 AND sales = 0 THEN 1 ELSE 0 END)            AS promo_sans_vente
FROM train;

-- Magasins hors plage attendue (1 à 54 pour ce jeu de données)
SELECT
    'magasins_hors_plage' AS controle,
    store_nbr,
    COUNT(*) AS nb_lignes
FROM train
WHERE store_nbr < 1 OR store_nbr > 54
GROUP BY store_nbr
ORDER BY store_nbr;

-- Familles inconnues ou mal formatées (espaces superflus)
SELECT
    family,
    COUNT(*) AS nb
FROM train
WHERE family <> TRIM(family)
GROUP BY family;


-- ------------------------------------------------------------
-- 4. VALEURS ABERRANTES (OUTLIERS)
-- ------------------------------------------------------------

-- Statistiques descriptives sur sales
SELECT
    'stats_sales' AS controle,
    MIN(sales)  AS min_sales,
    MAX(sales)  AS max_sales,
    AVG(sales)  AS avg_sales,
    STDDEV(sales) AS std_sales,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY sales) AS q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY sales) AS q3,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY sales) AS p99
FROM train;

-- Outliers méthode IQR (sur ventes strictement positives)
WITH stats AS (
    SELECT
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY sales) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY sales) AS q3
    FROM train
    WHERE sales > 0
),
bornes AS (
    SELECT
        q1,
        q3,
        q3 + 1.5 * (q3 - q1) AS limite_haute
    FROM stats
)
SELECT
    'outliers_iqr' AS controle,
    COUNT(*) AS nb_outliers,
    MIN(t.sales) AS sales_min_outlier,
    MAX(t.sales) AS sales_max_outlier
FROM train t
CROSS JOIN bornes b
WHERE t.sales > b.limite_haute;

-- Top 20 ventes les plus élevées (inspection manuelle)
SELECT
    id,
    date,
    store_nbr,
    family,
    sales,
    onpromotion
FROM train
ORDER BY sales DESC
LIMIT 20;

-- Outliers au-delà du 99e percentile
WITH p99 AS (
    SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY sales) AS seuil
    FROM train
)
SELECT
    'outliers_p99' AS controle,
    COUNT(*) AS nb_outliers,
    MAX(t.sales) AS sales_max
FROM train t
CROSS JOIN p99
WHERE t.sales > p99.seuil;

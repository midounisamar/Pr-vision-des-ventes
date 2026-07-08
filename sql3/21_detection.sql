-- ============================================================
-- 1 - DETECTION : valeurs manquantes, doublons,
--                  incohérences, valeurs aberrantes
-- Dataset : stores.csv (Store Sales - Kaggle)
-- Moteur  : DuckDB (compatible read_csv_auto)
-- ============================================================

CREATE OR REPLACE VIEW stores AS
SELECT * FROM read_csv_auto('stores.csv', header = true);


-- ------------------------------------------------------------
-- 1. VALEURS MANQUANTES (NULL)
-- ------------------------------------------------------------
SELECT
    'valeurs_manquantes' AS controle,
    COUNT(*)             AS total_lignes,
    SUM(CASE WHEN store_nbr IS NULL THEN 1 ELSE 0 END) AS store_nbr_null,
    SUM(CASE WHEN city IS NULL THEN 1 ELSE 0 END)      AS city_null,
    SUM(CASE WHEN state IS NULL THEN 1 ELSE 0 END)     AS state_null,
    SUM(CASE WHEN type IS NULL THEN 1 ELSE 0 END)      AS type_null,
    SUM(CASE WHEN cluster IS NULL THEN 1 ELSE 0 END)   AS cluster_null
FROM stores;


-- Détail par colonne (format long)
SELECT
    col AS colonne,
    nb_nulls,
    ROUND(100.0 * nb_nulls / total, 4) AS pct_null
FROM (
    SELECT COUNT(*) AS total FROM stores
) t
CROSS JOIN (
    SELECT 'store_nbr' AS col, SUM(CASE WHEN store_nbr IS NULL THEN 1 ELSE 0 END) AS nb_nulls FROM stores UNION ALL
    SELECT 'city',              SUM(CASE WHEN city IS NULL THEN 1 ELSE 0 END)              FROM stores UNION ALL
    SELECT 'state',             SUM(CASE WHEN state IS NULL THEN 1 ELSE 0 END)             FROM stores UNION ALL
    SELECT 'type',              SUM(CASE WHEN type IS NULL THEN 1 ELSE 0 END)              FROM stores UNION ALL
    SELECT 'cluster',           SUM(CASE WHEN cluster IS NULL THEN 1 ELSE 0 END)           FROM stores
) v;


-- ------------------------------------------------------------
-- 2. DOUBLONS
-- ------------------------------------------------------------

-- Doublons sur la clé technique (store_nbr)
SELECT
    'doublons_store_nbr' AS controle,
    COUNT(*) - COUNT(DISTINCT store_nbr) AS nb_doublons
FROM stores;

-- Doublons sur la clé métier complète (store_nbr + city + state + type + cluster)
SELECT
    'doublons_cle_metier' AS controle,
    COUNT(*) - COUNT(DISTINCT (store_nbr, city, state, type, cluster)) AS nb_doublons_metier
FROM stores;

-- Exemples de lignes dupliquées (store_nbr)
SELECT
    store_nbr,
    COUNT(*) AS occurrences
FROM stores
GROUP BY store_nbr
HAVING COUNT(*) > 1
ORDER BY occurrences DESC
LIMIT 20;


-- ------------------------------------------------------------
-- 3. INCOHÉRENCES
-- ------------------------------------------------------------
SELECT
    'incoherences' AS controle,
    SUM(CASE WHEN TRY_CAST(store_nbr AS INTEGER) IS NULL THEN 1 ELSE 0 END)              AS store_nbr_invalide,
    SUM(CASE WHEN store_nbr <= 0 OR store_nbr > 54 THEN 1 ELSE 0 END)                    AS store_nbr_hors_plage,
    SUM(CASE WHEN city IS NULL OR TRIM(city) = '' THEN 1 ELSE 0 END)                      AS city_vide,
    SUM(CASE WHEN state IS NULL OR TRIM(state) = '' THEN 1 ELSE 0 END)                    AS state_vide,
    SUM(CASE WHEN type IS NULL OR TRIM(type) = '' THEN 1 ELSE 0 END)                      AS type_vide,
    SUM(CASE WHEN cluster IS NULL THEN 1 ELSE 0 END)                                      AS cluster_null,
    SUM(CASE WHEN type NOT IN ('A', 'B', 'C', 'D', 'E') THEN 1 ELSE 0 END)                AS type_inconnu,
    SUM(CASE WHEN cluster < 1 OR cluster > 17 THEN 1 ELSE 0 END)                          AS cluster_hors_plage
FROM stores;


-- type mal formatés (espaces superflus)
SELECT
    type,
    COUNT(*) AS nb
FROM stores
WHERE type <> TRIM(type)
GROUP BY type;


-- city mal formatées (espaces superflus)
SELECT
    city,
    COUNT(*) AS nb
FROM stores
WHERE city <> TRIM(city)
GROUP BY city;


-- state mal formatées (espaces superflus)
SELECT
    state,
    COUNT(*) AS nb
FROM stores
WHERE state <> TRIM(state)
GROUP BY state;


-- ------------------------------------------------------------
-- 4. VALEURS ABERRANTES (OUTLIERS)
-- ------------------------------------------------------------

-- store_nbr hors plage attendue du projet (1 à 54)
SELECT
    'store_nbr_hors_plage' AS controle,
    store_nbr,
    city,
    state,
    type,
    cluster
FROM stores
WHERE store_nbr < 1 OR store_nbr > 54
ORDER BY store_nbr;


-- Répartition des types (détection de catégories rares)
SELECT
    'repartition_type' AS controle,
    type,
    COUNT(*) AS nb,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM stores
GROUP BY type
ORDER BY nb;


-- Répartition des villes
SELECT
    'repartition_city' AS controle,
    city,
    COUNT(*) AS nb,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM stores
GROUP BY city
ORDER BY nb;


-- Répartition des clusters
SELECT
    'repartition_cluster' AS controle,
    cluster,
    COUNT(*) AS nb,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM stores
GROUP BY cluster
ORDER BY cluster;


-- store_nbr associés à plusieurs villes (incohérence logique)
SELECT
    'store_nbr_multi_city' AS controle,
    store_nbr,
    COUNT(DISTINCT city) AS nb_villes_distinctes,
    COUNT(*) AS nb_lignes
FROM stores
GROUP BY store_nbr
HAVING COUNT(DISTINCT city) > 1
ORDER BY nb_villes_distinctes DESC
LIMIT 20;


-- Clusters rares (moins de 2 magasins)
SELECT
    'clusters_rares' AS controle,
    cluster,
    COUNT(*) AS nb_magasins
FROM stores
GROUP BY cluster
HAVING COUNT(*) < 2
ORDER BY nb_magasins;

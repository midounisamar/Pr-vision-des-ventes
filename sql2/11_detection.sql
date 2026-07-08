-- ============================================================
-- 1 - DETECTION : valeurs manquantes, doublons,
--                  incohérences, valeurs aberrantes
-- Dataset : holidays_events.csv (Store Sales - Kaggle)
-- Moteur  : DuckDB (compatible read_csv_auto)
-- ============================================================

CREATE OR REPLACE VIEW holidays_events AS
SELECT * FROM read_csv_auto('holidays_events.csv', header = true);


-- ------------------------------------------------------------
-- 1. VALEURS MANQUANTES (NULL)
-- ------------------------------------------------------------
SELECT
    'valeurs_manquantes' AS controle,
    COUNT(*)             AS total_lignes,
    SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END)           AS date_null,
    SUM(CASE WHEN type IS NULL THEN 1 ELSE 0 END)           AS type_null,
    SUM(CASE WHEN locale IS NULL THEN 1 ELSE 0 END)       AS locale_null,
    SUM(CASE WHEN locale_name IS NULL THEN 1 ELSE 0 END)  AS locale_name_null,
    SUM(CASE WHEN description IS NULL THEN 1 ELSE 0 END)  AS description_null,
    SUM(CASE WHEN transferred IS NULL THEN 1 ELSE 0 END)  AS transferred_null
FROM holidays_events;


-- Détail par colonne (format long)
SELECT
    col AS colonne,
    nb_nulls,
    ROUND(100.0 * nb_nulls / total, 4) AS pct_null
FROM (
    SELECT COUNT(*) AS total FROM holidays_events
) t
CROSS JOIN (
    SELECT 'date'        AS col, SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END)        AS nb_nulls FROM holidays_events UNION ALL
    SELECT 'type',                 SUM(CASE WHEN type IS NULL THEN 1 ELSE 0 END)                   FROM holidays_events UNION ALL
    SELECT 'locale',               SUM(CASE WHEN locale IS NULL THEN 1 ELSE 0 END)                 FROM holidays_events UNION ALL
    SELECT 'locale_name',          SUM(CASE WHEN locale_name IS NULL THEN 1 ELSE 0 END)            FROM holidays_events UNION ALL
    SELECT 'description',          SUM(CASE WHEN description IS NULL THEN 1 ELSE 0 END)            FROM holidays_events UNION ALL
    SELECT 'transferred',          SUM(CASE WHEN transferred IS NULL THEN 1 ELSE 0 END)            FROM holidays_events
) v;


-- ------------------------------------------------------------
-- 2. DOUBLONS
-- ------------------------------------------------------------

-- Doublons sur la clé événement (date + type + locale + locale_name + description + transferred)
SELECT
    'doublons_cle_evenement' AS controle,
    COUNT(*) - COUNT(DISTINCT (date, type, locale, locale_name, description, transferred)) AS nb_doublons_metier
FROM holidays_events;

-- Exemples de lignes dupliquées (clé événement)
SELECT
    date,
    type,
    locale,
    locale_name,
    description,
    transferred,
    COUNT(*) AS occurrences
FROM holidays_events
GROUP BY date, type, locale, locale_name, description, transferred
HAVING COUNT(*) > 1
ORDER BY occurrences DESC
LIMIT 20;


-- ------------------------------------------------------------
-- 3. INCOHÉRENCES
-- ------------------------------------------------------------
SELECT
    'incoherences' AS controle,
    SUM(CASE WHEN TRY_CAST(date AS DATE) IS NULL THEN 1 ELSE 0 END)                         AS date_invalide,
    SUM(CASE WHEN type IS NULL OR TRIM(type) = '' THEN 1 ELSE 0 END)                       AS type_vide,
    SUM(CASE WHEN locale IS NULL OR TRIM(locale) = '' THEN 1 ELSE 0 END)                   AS locale_vide,
    SUM(CASE WHEN locale_name IS NULL OR TRIM(locale_name) = '' THEN 1 ELSE 0 END)         AS locale_name_vide,
    SUM(CASE WHEN description IS NULL OR TRIM(description) = '' THEN 1 ELSE 0 END)         AS description_vide,
    SUM(CASE WHEN transferred IS NULL THEN 1 ELSE 0 END)                                     AS transferred_null,
    SUM(CASE WHEN type NOT IN ('Holiday', 'Transfer', 'Additional', 'Bridge', 'Work Day', 'Event') THEN 1 ELSE 0 END)
                                                                                             AS type_inconnu,
    SUM(CASE WHEN locale NOT IN ('Local', 'Regional', 'National') THEN 1 ELSE 0 END)       AS locale_inconnu,
    SUM(CASE WHEN locale = 'National' AND locale_name <> 'Ecuador' THEN 1 ELSE 0 END)        AS national_sans_ecuador,
    SUM(CASE WHEN locale = 'Local' AND locale_name = 'Ecuador' THEN 1 ELSE 0 END)            AS local_avec_ecuador
FROM holidays_events;


-- type inconnues ou mal formatées (espaces superflus)
SELECT
    type,
    COUNT(*) AS nb
FROM holidays_events
WHERE type <> TRIM(type)
GROUP BY type;


-- locale inconnues ou mal formatées (espaces superflus)
SELECT
    locale,
    COUNT(*) AS nb
FROM holidays_events
WHERE locale <> TRIM(locale)
GROUP BY locale;


-- locale_name mal formatées (espaces superflus)
SELECT
    locale_name,
    COUNT(*) AS nb
FROM holidays_events
WHERE locale_name <> TRIM(locale_name)
GROUP BY locale_name;


-- description mal formatées (espaces superflus)
SELECT
    description,
    COUNT(*) AS nb
FROM holidays_events
WHERE description <> TRIM(description)
GROUP BY description;


-- ------------------------------------------------------------
-- 4. VALEURS ABERRANTES (OUTLIERS)
-- ------------------------------------------------------------

-- Dates hors plage attendue du projet (2012-01-01 à 2017-12-31)
SELECT
    'dates_hors_plage' AS controle,
    COUNT(*) AS nb_lignes,
    MIN(date) AS date_min,
    MAX(date) AS date_max
FROM holidays_events
WHERE TRY_CAST(date AS DATE) < DATE '2012-01-01'
   OR TRY_CAST(date AS DATE) > DATE '2017-12-31';


-- Répartition des types (détection de catégories rares)
SELECT
    'repartition_type' AS controle,
    type,
    COUNT(*) AS nb,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM holidays_events
GROUP BY type
ORDER BY nb;


-- Répartition des locales
SELECT
    'repartition_locale' AS controle,
    locale,
    COUNT(*) AS nb,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM holidays_events
GROUP BY locale
ORDER BY nb;


-- Dates avec un nombre anormalement élevé d'événements (> 3, seuil empirique)
WITH events_par_date AS (
    SELECT
        date,
        COUNT(*) AS nb_events
    FROM holidays_events
    GROUP BY date
),
seuil AS (
    SELECT
        AVG(nb_events) + 2 * STDDEV(nb_events) AS limite
    FROM events_par_date
)
SELECT
    'dates_charge_anormale' AS controle,
    e.date,
    e.nb_events
FROM events_par_date e
CROSS JOIN seuil s
WHERE e.nb_events > s.limite
ORDER BY e.nb_events DESC;


-- Descriptions identiques sur des dates différentes (possible doublon logique)
SELECT
    'description_multi_dates' AS controle,
    description,
    COUNT(DISTINCT date) AS nb_dates_distinctes,
    COUNT(*) AS nb_lignes
FROM holidays_events
GROUP BY description
HAVING COUNT(DISTINCT date) > 1
ORDER BY nb_dates_distinctes DESC
LIMIT 20;


-- Événements transferred = TRUE (inspection des fêtes déplacées)
SELECT
    'evenements_transferes' AS controle,
    date,
    type,
    locale,
    locale_name,
    description,
    transferred
FROM holidays_events
WHERE transferred = TRUE
ORDER BY date;

-- ============================================================
-- 02 - NETTOYAGE des données holidays_events.csv
-- Moteur : DuckDB
-- Sortie : holidays_events_clean.csv
-- ============================================================

CREATE OR REPLACE VIEW holidays_events_raw AS
SELECT * FROM read_csv_auto('holidays_events.csv', header = true);


-- ------------------------------------------------------------
-- Étape 1 : normalisation et correction des incohérences
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW holidays_events_step1 AS
SELECT
    CAST(date AS DATE)      AS date,
    TRIM(type)              AS type,
    TRIM(locale)            AS locale,
    TRIM(locale_name)       AS locale_name,
    TRIM(description)       AS description,
    transferred
FROM holidays_events_raw
WHERE date IS NOT NULL
  AND type IS NOT NULL
  AND locale IS NOT NULL
  AND locale_name IS NOT NULL
  AND description IS NOT NULL
  AND transferred IS NOT NULL
  AND TRIM(type) <> ''
  AND TRIM(locale) <> ''
  AND TRIM(locale_name) <> ''
  AND TRIM(description) <> ''
  AND TRY_CAST(date AS DATE) IS NOT NULL
  AND type IN ('Holiday', 'Transfer', 'Additional', 'Bridge', 'Work Day', 'Event')
  AND locale IN ('Local', 'Regional', 'National');


-- ------------------------------------------------------------
-- Étape 2 : suppression des doublons (garder la 1re occurrence)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW holidays_events_step2 AS
SELECT
    date,
    type,
    locale,
    locale_name,
    description,
    transferred
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY date, type, locale, locale_name, description, transferred
            ORDER BY date, type, locale, locale_name, description, transferred
        ) AS rn
    FROM holidays_events_step1
) dedup
WHERE rn = 1;


-- ------------------------------------------------------------
-- Étape 3 : filtrage des valeurs aberrantes
-- (dates hors plage du projet : 2012-01-01 à 2017-12-31)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW holidays_events_clean AS
SELECT
    date,
    type,
    locale,
    locale_name,
    description,
    transferred,
    CASE
        WHEN date < DATE '2012-01-01' OR date > DATE '2017-12-31' THEN TRUE
        ELSE FALSE
    END AS date_hors_plage
FROM holidays_events_step2
WHERE date BETWEEN DATE '2012-01-01' AND DATE '2017-12-31';


-- ------------------------------------------------------------
-- Export du fichier nettoyé (sans les colonnes techniques)
-- ------------------------------------------------------------
COPY (
    SELECT date, type, locale, locale_name, description, transferred
    FROM holidays_events_clean
    ORDER BY date, type, locale, locale_name
) TO 'holidays_events_clean.csv' (HEADER, DELIMITER ',');

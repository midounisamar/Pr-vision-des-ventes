"""
Modèle SARIMAX pour la prévision des ventes avec toutes les étapes requises.

Usage:
    pip install pmdarima
    py -3 sarimax_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)


def load_and_prepare_data():
    """Charger et préparer les données pour le modèle SARIMAX."""
    print("=" * 60)
    print("CHARGEMENT ET PRÉPARATION DES DONNÉES")
    print("=" * 60)
    
    # Charger les données
    train_df = pd.read_csv('train_clean.csv')
    holidays_df = pd.read_csv('holidays_events_clean.csv')
    stores_df = pd.read_csv('stores_clean.csv')
    
    # Convertir la date
    train_df['date'] = pd.to_datetime(train_df['date'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    
    # Agréger les ventes par date
    daily_sales = train_df.groupby('date').agg({
        'sales': 'sum',
        'onpromotion': 'sum'
    }).reset_index()
    
    # Créer des variables exogènes
    daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['year'] = daily_sales['date'].dt.year
    daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
    
    # Ajouter les jours fériés
    holidays_df['is_holiday'] = 1
    holiday_dates = holidays_df[['date', 'is_holiday']].drop_duplicates()
    daily_sales = daily_sales.merge(holiday_dates, on='date', how='left')
    daily_sales['is_holiday'] = daily_sales['is_holiday'].fillna(0)
    
    # Définir l'index temporel
    daily_sales = daily_sales.set_index('date')
    daily_sales = daily_sales.asfreq('D')
    
    # Remplir les valeurs manquantes
    daily_sales['sales'] = daily_sales['sales'].fillna(0)
    daily_sales['onpromotion'] = daily_sales['onpromotion'].fillna(0)
    daily_sales['is_holiday'] = daily_sales['is_holiday'].fillna(0)
    
    print(f"Dimensions des données: {daily_sales.shape}")
    print(f"Période: {daily_sales.index.min()} à {daily_sales.index.max()}")
    print(f"\nPremières lignes:")
    print(daily_sales.head())
    
    return daily_sales


def test_stationarity(series, title="Série temporelle"):
    """Tester la stationnarité avec les tests ADF et KPSS."""
    print("\n" + "=" * 60)
    print(f"TESTS DE STATIONNARITÉ - {title}")
    print("=" * 60)
    
    # Test ADF (Augmented Dickey-Fuller)
    print("\n--- Test ADF (Augmented Dickey-Fuller) ---")
    adf_result = adfuller(series.dropna(), regression='ct')
    print(f"Statistique ADF: {adf_result[0]:.4f}")
    print(f"P-value: {adf_result[1]:.4f}")
    print(f"Valeurs critiques:")
    for key, value in adf_result[4].items():
        print(f"  {key}: {value:.4f}")
    
    if adf_result[1] < 0.05:
        print("[OK] La serie est STATIONNAIRE (rejet de H0)")
    else:
        print("[X] La serie est NON-STATIONNAIRE (acceptation de H0)")
    
    # Test KPSS
    print("\n--- Test KPSS ---")
    kpss_result = kpss(series.dropna(), regression='ct', nlags='auto')
    print(f"Statistique KPSS: {kpss_result[0]:.4f}")
    print(f"P-value: {kpss_result[1]:.4f}")
    print(f"Valeurs critiques:")
    for key, value in kpss_result[3].items():
        print(f"  {key}: {value:.4f}")
    
    if kpss_result[1] > 0.05:
        print("[OK] La serie est STATIONNAIRE (acceptation de H0)")
    else:
        print("[X] La serie est NON-STATIONNAIRE (rejet de H0)")
    
    return adf_result[1] < 0.05 and kpss_result[1] > 0.05


def apply_differencing(series, max_diff=2):
    """Appliquer la différenciation si nécessaire et tester la stationnarité."""
    print("\n" + "=" * 60)
    print("DIFFÉRENCIATION")
    print("=" * 60)
    
    is_stationary = test_stationarity(series, "Série originale")
    
    if is_stationary:
        print("\n[OK] La serie est deja stationnaire, pas de differenciation necessaire.")
        return series, 0
    
    diff_order = 0
    diff_series = series.copy()
    
    for d in range(1, max_diff + 1):
        diff_order = d
        diff_series = diff_series.diff().dropna()
        print(f"\n--- Après différenciation d'ordre {d} ---")
        is_stationary = test_stationarity(diff_series, f"Série différenciée (d={d})")
        
        if is_stationary:
            print(f"\n[OK] Stationnarité atteinte avec d={d}")
            return diff_series, d
    
    print(f"\n[!] Stationnarite non atteinte apres {max_diff} differentiations.")
    return diff_series, diff_order


def plot_acf_pacf(series, lags=40):
    """Afficher les graphiques ACF et PACF pour déterminer p et q."""
    print("\n" + "=" * 60)
    print("GRAPHIQUES ACF ET PACF")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF
    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('ACF - Autocorrélation')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Corrélation')
    
    # PACF
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title('PACF - Autocorrélation Partielle')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Corrélation')
    
    plt.tight_layout()
    plt.savefig('eda_output/acf_pacf.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphiques ACF/PACF sauvegardes dans eda_output/acf_pacf.png")
    plt.close()
    
    print("\nInterprétation pour les paramètres p et q:")
    print("- p (AR): Regarder le PACF - nombre de lags significatifs avant coupure")
    print("- q (MA): Regarder l'ACF - nombre de lags significatifs avant coupure")


def determine_seasonal_params(series, period=7):
    """Déterminer les paramètres saisonniers P, D, Q, s."""
    print("\n" + "=" * 60)
    print("DÉTERMINATION DES PARAMÈTRES SAISONNIERS")
    print("=" * 60)
    
    print(f"\nPériode saisonnière s: {period} (hebdomadaire)")
    
    # Tester la stationnarité saisonnière
    seasonal_diff = series.diff(period).dropna()
    is_seasonal_stationary = test_stationarity(seasonal_diff, "Différenciation saisonnière")
    
    if is_seasonal_stationary:
        D = 1
        print(f"\n[OK] Differentiation saisonniere necessaire: D=1")
    else:
        D = 0
        print(f"\n[X] Pas de differentiation saisonniere necessaire: D=0")
    
    # Analyse ACF/PACF saisonnière
    print("\n--- Analyse ACF/PACF aux lags multiples de s ---")
    print(f"Regarder les lags {period}, {2*period}, {3*period}, etc. dans ACF et PACF")
    print(f"P: nombre de pics significatifs aux lags multiples de s dans le PACF")
    print(f"Q: nombre de pics significatifs aux lags multiples de s dans l'ACF")
    
    return D, period


def train_sarimax_model(endog, exog, order, seasonal_order):
    """Entraîner le modèle SARIMAX."""
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE SARIMAX")
    print("=" * 60)
    
    print(f"\nParamètres ARIMA (p,d,q): {order}")
    print(f"Paramètres saisonniers (P,D,Q,s): {seasonal_order}")
    print(f"Variables exogènes: {exog.columns.tolist() if exog is not None else 'Aucune'}")
    
    try:
        model = SARIMAX(
            endog=endog,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        print("\n[...] Entrainement en cours...")
        results = model.fit(disp=False)
        print("[OK] Modele entraine avec succes")
        
        print(f"\n--- Résumé du modèle ---")
        print(results.summary().tables[1])
        
        return results
    except Exception as e:
        print(f"\n[X] Erreur lors de l'entrainement: {e}")
        return None


def residual_diagnostics(results):
    """Effectuer le diagnostic des résidus."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC DES RÉSIDUS")
    print("=" * 60)
    
    residuals = results.resid
    
    # Test de Ljung-Box
    print("\n--- Test de Ljung-Box (autocorrélation des résidus) ---")
    ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(ljung_box)
    
    if ljung_box['lb_pvalue'].iloc[0] > 0.05:
        print("[OK] Residus non autocorrelés (p-value > 0.05)")
    else:
        print("[X] Residus autocorrelés (p-value <= 0.05)")
    
    # Test de normalité
    print("\n--- Test de normalité (Jarque-Bera) ---")
    jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
    print(f"Statistique Jarque-Bera: {jb_stat:.4f}")
    print(f"P-value: {jb_pvalue:.4f}")
    
    if jb_pvalue > 0.05:
        print("[OK] Residus normaux (p-value > 0.05)")
    else:
        print("[X] Residus non normaux (p-value <= 0.05)")
    
    # Graphiques des résidus
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Série des résidus
    axes[0, 0].plot(residuals)
    axes[0, 0].set_title('Résidus dans le temps')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    
    # Histogramme
    axes[0, 1].hist(residuals.dropna(), bins=30, edgecolor='black')
    axes[0, 1].set_title('Distribution des résidus')
    axes[0, 1].set_xlabel('Valeur')
    axes[0, 1].set_ylabel('Fréquence')
    
    # Q-Q plot
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # ACF des résidus
    plot_acf(residuals.dropna(), lags=20, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('ACF des résidus')
    
    plt.tight_layout()
    plt.savefig('eda_output/residual_diagnostics.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Graphiques de diagnostic sauvegardes dans eda_output/residual_diagnostics.png")
    plt.close()
    
    return residuals


def generate_forecasts_and_evaluate(results, endog, exog, n_steps=30):
    """Générer les prévisions et évaluer le modèle."""
    print("\n" + "=" * 60)
    print("PRÉVISIONS ET ÉVALUATION")
    print("=" * 60)
    
    # Diviser en train/test
    train_size = len(endog) - n_steps
    y_train = endog[:train_size]
    y_test = endog[train_size:]
    
    if exog is not None:
        exog_train = exog[:train_size]
        exog_test = exog[train_size:]
    else:
        exog_train = None
        exog_test = None
    
    # Réentraîner sur les données d'entraînement
    print(f"\nTaille entraînement: {len(y_train)}")
    print(f"Taille test: {len(y_test)}")
    
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=results.specification.order,
        seasonal_order=results.specification.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)
    
    # Prévisions
    print("\n[...]..e]eGeneration dee previsions...")
    forecast = fitted_model.forecast(steps=n_steps, exog=exog_test)
    
    # Calculer les métriques
    mae = mean_absolute_error(y_test, forecast)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    
    # MAPE (éviter la division par zéro)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - forecast[mask]) / y_test[mask])) * 100
    
    print(f"\n--- Métriques d'évaluation ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # Graphique des prévisions
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Données historiques
    ax.plot(y_train.index, y_train, label='Entraînement', alpha=0.7)
    ax.plot(y_test.index, y_test, label='Réel (test)', color='green')
    ax.plot(forecast.index, forecast, label='Prévisions', color='red', linestyle='--')
    
    ax.set_title('Prévisions SARIMAX vs Réalité')
    ax.set_xlabel('Date')
    ax.set_ylabel('Ventes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_output/forecasts.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Graphique des previsions sauvegarde dans eda_output/forecasts.png")
    plt.close()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'forecast': forecast,
        'y_test': y_test
    }


def plot_evaluation_metrics(mae, rmse, mape):
    """Créer un graphique des métriques d'évaluation."""
    print("\n" + "=" * 60)
    print("VISUALISATION DES MÉTRIQUES D'ÉVALUATION")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['MAE', 'RMSE', 'MAPE (%)']
    values = [mae, rmse, mape]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
    
    # Ajouter les valeurs au-dessus des barres
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Métriques d\'Évaluation du Modèle SARIMAX', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valeur', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('eda_output/evaluation_metrics.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphique des métriques sauvegardé dans eda_output/evaluation_metrics.png")
    plt.close()


def main():
    """Fonction principale exécutant toutes les étapes."""
    from pathlib import Path
    
    # Créer le dossier de sortie
    Path('eda_output').mkdir(exist_ok=True)
    
    # 1. Charger et préparer les données
    data = load_and_prepare_data()
    
    # Séparer la variable cible et les variables exogènes
    endog = data['sales']
    exog = data[['onpromotion', 'day_of_week', 'month', 'is_weekend', 'is_holiday']]
    
    # Nettoyer les variables exogènes (remplacer inf et NaN)
    exog = exog.replace([np.inf, -np.inf], np.nan)
    exog = exog.fillna(0)
    
    # 2. Tester la stationnarité
    is_stationary = test_stationarity(endog, "Ventes quotidiennes")
    
    # 3. Appliquer la différenciation si nécessaire
    if not is_stationary:
        diff_series, d = apply_differencing(endog, max_diff=2)
    else:
        diff_series, d = endog, 0
    
    # 4. Afficher ACF et PACF
    plot_acf_pacf(diff_series, lags=40)
    
    # 5. Déterminer les paramètres saisonniers
    D, s = determine_seasonal_params(endog, period=7)
    
    # Paramètres suggérés (à ajuster selon ACF/PACF)
    p = 1  # À ajuster selon PACF
    q = 1  # À ajuster selon ACF
    P = 1  # À ajuster selon PACF saisonnier
    Q = 1  # À ajuster selon ACF saisonnier
    
    order = (p, d, q)
    seasonal_order = (P, D, Q, s)
    
    print(f"\n--- Paramètres choisis ---")
    print(f"ARIMA (p,d,q): {order}")
    print(f"Saisonnier (P,D,Q,s): {seasonal_order}")
    
    # 6. Entraîner le modèle SARIMAX
    results = train_sarimax_model(endog, exog, order, seasonal_order)
    
    if results is None:
        print("\n[X] Echec de l'entrainement du modele")
        return
    
    # 7. Diagnostic des résidus
    residual_diagnostics(results)
    
    # 8. Prévisions et évaluation
    evaluation = generate_forecasts_and_evaluate(results, endog, exog, n_steps=30)
    
    # 9. Visualisation des métriques d'évaluation
    plot_evaluation_metrics(evaluation['mae'], evaluation['rmse'], evaluation['mape'])
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"\nParamètres du modèle: ARIMA{order} x {seasonal_order}")
    print(f"\nMétriques sur les {len(evaluation['y_test'])} derniers jours:")
    print(f"  MAE:  {evaluation['mae']:.2f}")
    print(f"  RMSE: {evaluation['rmse']:.2f}")
    print(f"  MAPE: {evaluation['mape']:.2f}%")
    print("\n[OK] Analyse SARIMAX terminee avec succes")
    print("[OK] Graphiques sauvegardes dans le dossier eda_output/")


if __name__ == "__main__":
    main()

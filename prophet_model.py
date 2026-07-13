"""
Modèle Prophet pour la prévision des ventes avec toutes les étapes requises.

Usage:
    pip install prophet
    py -3 prophet_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)


def load_and_prepare_data():
    """Charger et préparer les données pour le modèle Prophet."""
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
    
    # Créer des variables explicatives
    daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['year'] = daily_sales['date'].dt.year
    daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
    
    # Ajouter les jours fériés
    holidays_df['is_holiday'] = 1
    holiday_dates = holidays_df[['date', 'is_holiday']].drop_duplicates()
    daily_sales = daily_sales.merge(holiday_dates, on='date', how='left')
    daily_sales['is_holiday'] = daily_sales['is_holiday'].fillna(0)
    
    # Préparer au format Prophet (ds, y)
    prophet_df = daily_sales.rename(columns={'date': 'ds', 'sales': 'y'})
    
    print(f"Dimensions des données: {prophet_df.shape}")
    print(f"Période: {prophet_df['ds'].min()} à {prophet_df['ds'].max()}")
    print(f"\nPremières lignes:")
    print(prophet_df.head())
    
    return prophet_df, holidays_df


def prepare_holidays(holidays_df):
    """Préparer les jours fériés pour Prophet."""
    print("\n" + "=" * 60)
    print("PRÉPARATION DES JOURS FÉRIÉS")
    print("=" * 60)
    
    # Créer un dataframe au format Prophet
    holidays = holidays_df[['date', 'description', 'type']].copy()
    holidays = holidays.rename(columns={'date': 'ds', 'description': 'holiday', 'type': 'type'})
    
    # Garder uniquement les colonnes nécessaires
    holidays = holidays[['ds', 'holiday']]
    
    # Supprimer les doublons
    holidays = holidays.drop_duplicates(subset=['ds'])
    
    print(f"Nombre de jours fériés: {len(holidays)}")
    print(f"\nExemples de jours fériés:")
    print(holidays.head(10))
    
    return holidays


def train_prophet_model(df, holidays, regressors):
    """Entraîner le modèle Prophet avec régresseurs et jours fériés."""
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE PROPHET")
    print("=" * 60)
    
    print(f"\nVariables explicatives: {regressors}")
    print(f"Jours fériés: {len(holidays)}")
    
    # Créer le modèle Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        holidays_prior_scale=10
    )
    
    # Ajouter les régresseurs
    for regressor in regressors:
        model.add_regressor(regressor)
        print(f"  - Régresseur ajouté: {regressor}")
    
    # Entraîner le modèle
    print("\n[...] Entraînement en cours...")
    model.fit(df)
    print("[OK] Modèle entraîné avec succès")
    
    return model


def generate_forecasts(model, df, periods=30):
    """Générer les prévisions avec le modèle Prophet."""
    print("\n" + "=" * 60)
    print("GÉNÉRATION DES PRÉVISIONS")
    print("=" * 60)
    
    # Créer le dataframe futur avec les régresseurs
    future = model.make_future_dataframe(periods=periods, freq='D')
    
    # Ajouter les régresseurs au dataframe futur
    # Pour simplifier, on utilise les valeurs moyennes des régresseurs
    for col in ['onpromotion', 'day_of_week', 'month', 'is_weekend', 'is_holiday']:
        if col in df.columns:
            future[col] = future['ds'].dt.dayofweek if col == 'day_of_week' else \
                          future['ds'].dt.month if col == 'month' else \
                          (future['ds'].dt.dayofweek >= 5).astype(int) if col == 'is_weekend' else \
                          df[col].mean() if col in ['onpromotion', 'is_holiday'] else 0
    
    # Générer les prévisions
    print(f"[...] Génération des prévisions pour {periods} jours...")
    forecast = model.predict(future)
    print("[OK] Prévisions générées")
    
    print(f"\nDimensions des prévisions: {forecast.shape}")
    print(f"\nColonnes disponibles: {forecast.columns.tolist()}")
    
    return forecast


def evaluate_forecasts(forecast, df, periods=30):
    """Évaluer les prévisions avec MAE, RMSE et MAPE."""
    print("\n" + "=" * 60)
    print("ÉVALUATION DES PRÉVISIONS")
    print("=" * 60)
    
    # Extraire les prévisions pour la période de test
    forecast_test = forecast.tail(periods)
    
    # Extraire les valeurs réelles correspondantes
    y_test = df.tail(periods)['y'].values
    y_pred = forecast_test['yhat'].values
    
    # Calculer les métriques
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # MAPE (éviter la division par zéro)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    
    print(f"\n--- Métriques d'évaluation ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'y_test': y_test,
        'y_pred': y_pred,
        'forecast_test': forecast_test
    }


def plot_prophet_forecast(model, forecast, df):
    """Créer les visualisations Prophet."""
    print("\n" + "=" * 60)
    print("VISUALISATIONS PROPHET")
    print("=" * 60)
    
    # 1. Graphique principal des prévisions
    fig1 = model.plot(forecast, figsize=(14, 7))
    plt.title('Prévisions Prophet - Ventes', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Ventes', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_output/prophet_forecast.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphique des prévisions sauvegardé dans eda_output/prophet_forecast.png")
    plt.close()
    
    # 2. Graphique des composantes
    fig2 = model.plot_components(forecast, figsize=(14, 10))
    plt.tight_layout()
    plt.savefig('eda_output/prophet_components.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphique des composantes sauvegardé dans eda_output/prophet_components.png")
    plt.close()
    
    # 3. Graphique comparatif prévisions vs réalité
    fig3, ax = plt.subplots(figsize=(14, 6))
    
    # Données historiques
    ax.plot(df['ds'], df['y'], label='Historique', alpha=0.7, color='blue')
    
    # Prévisions
    ax.plot(forecast['ds'], forecast['yhat'], label='Prévisions', color='red', linestyle='--')
    
    # Intervalle de confiance
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    alpha=0.2, color='red', label='Intervalle de confiance')
    
    ax.set_title('Prévisions Prophet vs Historique', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Ventes', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_output/prophet_comparison.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphique comparatif sauvegardé dans eda_output/prophet_comparison.png")
    plt.close()


def plot_evaluation_metrics_prophet(mae, rmse, mape):
    """Créer un graphique des métriques d'évaluation Prophet."""
    print("\n" + "=" * 60)
    print("VISUALISATION DES MÉTRIQUES D'ÉVALUATION")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['MAE', 'RMSE', 'MAPE (%)']
    values = [mae, rmse, mape]
    colors = ['#9b59b6', '#e67e22', '#1abc9c']
    
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
    
    # Ajouter les valeurs au-dessus des barres
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Métriques d\'Évaluation du Modèle Prophet', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valeur', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('eda_output/prophet_evaluation_metrics.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphique des métriques sauvegardé dans eda_output/prophet_evaluation_metrics.png")
    plt.close()


def main():
    """Fonction principale exécutant toutes les étapes."""
    from pathlib import Path
    
    # Créer le dossier de sortie
    Path('eda_output').mkdir(exist_ok=True)
    
    # 1. Charger et préparer les données
    df, holidays_df = load_and_prepare_data()
    
    # 2. Préparer les jours fériés
    holidays = prepare_holidays(holidays_df)
    
    # 3. Définir les régresseurs
    regressors = ['onpromotion', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
    
    # 4. Entraîner le modèle Prophet
    model = train_prophet_model(df, holidays, regressors)
    
    # 5. Générer les prévisions
    forecast = generate_forecasts(model, df, periods=30)
    
    # 6. Évaluer les prévisions
    evaluation = evaluate_forecasts(forecast, df, periods=30)
    
    # 7. Créer les visualisations
    plot_prophet_forecast(model, forecast, df)
    plot_evaluation_metrics_prophet(evaluation['mae'], evaluation['rmse'], evaluation['mape'])
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"\nModèle Prophet entraîné avec:")
    print(f"  - {len(holidays)} jours fériés")
    print(f"  - {len(regressors)} régresseurs: {regressors}")
    print(f"\nMétriques sur les 30 derniers jours:")
    print(f"  MAE:  {evaluation['mae']:.2f}")
    print(f"  RMSE: {evaluation['rmse']:.2f}")
    print(f"  MAPE: {evaluation['mape']:.2f}%")
    print("\n[OK] Analyse Prophet terminée avec succès")
    print("[OK] Graphiques sauvegardés dans le dossier eda_output/")


if __name__ == "__main__":
    main()

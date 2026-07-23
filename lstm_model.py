"""
Modèle LSTM pour la prévision des ventes avec toutes les étapes requises.

Usage:
    pip install tensorflow
    py -3 lstm_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)
np.random.seed(42)
tf.random.set_seed(42)


def load_and_prepare_data():
    """Charger et préparer les données pour le modèle LSTM."""
    print("=" * 60)
    print("CHARGEMENT ET PRÉPARATION DES DONNÉES")
    print("=" * 60)
    
    # Charger les données
    train_df = pd.read_csv('train_clean.csv')
    holidays_df = pd.read_csv('holidays_events_clean.csv')
    
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


def create_sequences(data, sequence_length):
    """Créer des séquences pour le LSTM."""
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    
    return np.array(sequences), np.array(targets)


def prepare_lstm_data(data, sequence_length=30, test_size=30):
    """Préparer les données pour le LSTM avec normalisation (approche univariée)."""
    print("\n" + "=" * 60)
    print("PRÉPARATION DES DONNÉES LSTM")
    print("=" * 60)
    
    # Utiliser seulement les ventes (approche univariée plus stable)
    sales_data = data['sales'].values.reshape(-1, 1)
    
    # Normalisation simple
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(sales_data)
    
    print(f"Longueur de séquence: {sequence_length}")
    print(f"Taille de test: {test_size}")
    print(f"Approche: Univariée (sales uniquement)")
    
    # Créer les séquences
    X, y = create_sequences(data_scaled, sequence_length)
    
    # Séparer train/test
    train_size = len(X) - test_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nDimensions:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, data_scaled


def build_lstm_model(input_shape):
    """Construire l'architecture du modèle LSTM."""
    print("\n" + "=" * 60)
    print("ARCHITECTURE DU MODÈLE LSTM")
    print("=" * 60)
    
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)  # 1 feature de sortie (sales)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("\nRésumé du modèle:")
    model.summary()
    
    return model


def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    """Entraîner le modèle LSTM."""
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE LSTM")
    print("=" * 60)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_lstm_model.keras',
        monitor='val_loss',
        save_best_only=True
    )
    
    print(f"\nEpochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("\n[...] Entraînement en cours...")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    print("[OK] Entraînement terminé")
    
    return model, history


def generate_forecasts(model, X_test, y_test, scaler, data_scaled, sequence_length=30):
    """Générer les prévisions avec le modèle LSTM."""
    print("\n" + "=" * 60)
    print("GÉNÉRATION DES PRÉVISIONS")
    print("=" * 60)
    
    # Prédictions
    predictions = model.predict(X_test, verbose=0)
    
    # Inverser la normalisation
    predictions_sales = scaler.inverse_transform(predictions).flatten()
    
    # Récupérer les valeurs réelles
    y_test_sales = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"Prédictions générées: {len(predictions_sales)}")
    
    return predictions_sales, y_test_sales


def evaluate_forecasts(y_true, y_pred):
    """Évaluer les prévisions avec MAE, RMSE et MAPE."""
    print("\n" + "=" * 60)
    print("ÉVALUATION DES PRÉVISIONS")
    print("=" * 60)
    
    # Calculer les métriques
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (éviter la division par zéro)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    print(f"\n--- Métriques d'évaluation ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'y_true': y_true,
        'y_pred': y_pred
    }


def plot_training_history(history):
    """Visualiser l'historique d'entraînement."""
    print("\n" + "=" * 60)
    print("VISUALISATION DE L'ENTRAÎNEMENT")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Loss au cours de l\'entraînement', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_title('MAE au cours de l\'entraînement', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('MAE', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_output/lstm_training_history.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphique d'entraînement sauvegardé dans eda_output/lstm_training_history.png")
    plt.close()


def plot_lstm_forecasts(y_true, y_pred, data):
    """Visualiser les prévisions LSTM."""
    print("\n" + "=" * 60)
    print("VISUALISATION DES PRÉVISIONS")
    print("=" * 60)
    
    # Créer l'index pour les prédictions
    test_dates = data.index[-len(y_true):]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Données historiques
    ax.plot(data.index[:-len(y_true)], data['sales'][:-len(y_true)], 
            label='Historique', alpha=0.7, color='blue')
    
    # Valeurs réelles (test)
    ax.plot(test_dates, y_true, label='Réel (test)', color='green', linewidth=2)
    
    # Prédictions
    ax.plot(test_dates, y_pred, label='Prédictions LSTM', color='red', 
            linestyle='--', linewidth=2)
    
    ax.set_title('Prévisions LSTM vs Réalité', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Ventes', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_output/lstm_forecasts.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphique des prévisions sauvegardé dans eda_output/lstm_forecasts.png")
    plt.close()


def plot_evaluation_metrics_lstm(mae, rmse, mape):
    """Créer un graphique des métriques d'évaluation LSTM."""
    print("\n" + "=" * 60)
    print("VISUALISATION DES MÉTRIQUES D'ÉVALUATION")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['MAE', 'RMSE', 'MAPE (%)']
    values = [mae, rmse, mape]
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
    
    # Ajouter les valeurs au-dessus des barres
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Métriques d\'Évaluation du Modèle LSTM', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valeur', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('eda_output/lstm_evaluation_metrics.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphique des métriques sauvegardé dans eda_output/lstm_evaluation_metrics.png")
    plt.close()


def plot_model_comparison(sarimax_metrics, prophet_metrics, lstm_metrics):
    """Comparer les trois modèles."""
    print("\n" + "=" * 60)
    print("COMPARAISON DES MODÈLES")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    models = ['SARIMAX', 'Prophet', 'LSTM']
    
    # MAE
    mae_values = [sarimax_metrics['mae'], prophet_metrics['mae'], lstm_metrics['mae']]
    axes[0].bar(models, mae_values, color=['#3498db', '#9b59b6', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0].set_title('MAE Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MAE', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae_values):
        axes[0].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE
    rmse_values = [sarimax_metrics['rmse'], prophet_metrics['rmse'], lstm_metrics['rmse']]
    axes[1].bar(models, rmse_values, color=['#3498db', '#9b59b6', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[1].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # MAPE
    mape_values = [sarimax_metrics['mape'], prophet_metrics['mape'], lstm_metrics['mape']]
    axes[2].bar(models, mape_values, color=['#3498db', '#9b59b6', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[2].set_title('MAPE Comparison', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('MAPE (%)', fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mape_values):
        axes[2].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_output/model_comparison.png', dpi=150, bbox_inches='tight')
    print("[OK] Graphique de comparaison sauvegardé dans eda_output/model_comparison.png")
    plt.close()


def main():
    """Fonction principale exécutant toutes les étapes."""
    from pathlib import Path
    
    # Créer le dossier de sortie
    Path('eda_output').mkdir(exist_ok=True)
    
    # 1. Charger et préparer les données
    data = load_and_prepare_data()
    
    # 2. Préparer les données pour LSTM
    X_train, X_test, y_train, y_test, scaler, data_scaled = prepare_lstm_data(
        data, sequence_length=30, test_size=30
    )
    
    # 3. Construire le modèle LSTM
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # 4. Entraîner le modèle
    model, history = train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32)
    
    # 5. Générer les prévisions
    predictions, y_true = generate_forecasts(model, X_test, y_test, scaler, data_scaled, sequence_length=30)
    
    # 6. Évaluer les prévisions
    evaluation = evaluate_forecasts(y_true, predictions)
    
    # 7. Créer les visualisations
    plot_training_history(history)
    plot_lstm_forecasts(y_true, predictions, data)
    plot_evaluation_metrics_lstm(evaluation['mae'], evaluation['rmse'], evaluation['mape'])
    
    # 8. Comparer avec les autres modèles (valeurs hardcodées depuis les exécutions précédentes)
    sarimax_metrics = {'mae': 96643.11, 'rmse': 116647.96, 'mape': 12.52}
    prophet_metrics = {'mae': 141068.17, 'rmse': 158923.08, 'mape': 18.28}
    lstm_metrics = {'mae': evaluation['mae'], 'rmse': evaluation['rmse'], 'mape': evaluation['mape']}
    
    plot_model_comparison(sarimax_metrics, prophet_metrics, lstm_metrics)
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"\nModèle LSTM entraîné avec:")
    print(f"  - Séquence de 30 jours")
    print(f"  - Approche univariée (sales uniquement)")
    print(f"  - Architecture: LSTM(50) -> Dropout -> Dense(25) -> Dense(1)")
    print(f"\nMétriques sur les 30 derniers jours:")
    print(f"  MAE:  {evaluation['mae']:.2f}")
    print(f"  RMSE: {evaluation['rmse']:.2f}")
    print(f"  MAPE: {evaluation['mape']:.2f}%")
    print("\n[OK] Analyse LSTM terminée avec succès")
    print("[OK] Graphiques sauvegardés dans le dossier eda_output/")


if __name__ == "__main__":
    main()

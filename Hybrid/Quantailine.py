import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask import Flask, request, jsonify
import os
import logging

# ============ Logging ============
logging.basicConfig(level=logging.INFO)

# ============ MetaTrader5 Setup ============
def connect_mt5():
    if not mt5.initialize(login=12345678, server="MetaQuotes-Demo", password="password123"):
        raise ConnectionError(f"Gagal koneksi MT5: {mt5.last_error()}")
    logging.info("✅ Berhasil login ke MetaTrader5")

def acquire_data(symbol, timeframe, start, end):
    connect_mt5()
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise ValueError("❌ Gagal mengambil data dari MT5")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# ============ Preprocessing ============
def preprocess(df):
    df = df.dropna().reset_index(drop=True)
    features = ['open','high','low','close','tick_volume']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def create_dataset(df, window_size):
    X, y = [], []
    for i in range(len(df)-window_size):
        window = df.iloc[i:i+window_size][['open','high','low','close','tick_volume']].values
        target = df.iloc[i+window_size][['high','low','close']].values
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

# ============ Model (LSTM + Attention) ============
def build_lstm_attention_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(output_dim)(x)
    return Model(inputs, outputs)

# ============ Training ============
def train_model(model, X_train, y_train, X_val, y_val, name):
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    checkpoint_path = f"{name}.h5"
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=50, batch_size=32, callbacks=callbacks, verbose=0)
    model.load_weights(checkpoint_path)
    return model

# ============ Evaluation ============
def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test, verbose=0)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    logging.info(f"📊 MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return pred

# ============ Main Training ============
def main_training():
    df = acquire_data('XAUUSD', mt5.TIMEFRAME_D1, pd.Timestamp(2008,1,1), pd.Timestamp(2014,12,31))
    df, scaler = preprocess(df)
    X, y = create_dataset(df, window_size=10)

    tscv = TimeSeriesSplit(n_splits=3)
    final_model = None
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logging.info(f"🔁 Fold {fold+1}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = build_lstm_attention_model(X.shape[1:], y.shape[1])
        final_model = train_model(model, X_train, y_train, X_val, y_val, name=f"model_fold{fold+1}")
        evaluate_model(final_model, X_val, y_val)

    return final_model

# ============ Flask API ============
app = Flask(__name__)
loaded_model = None

@app.before_first_request
def load_model():
    global loaded_model
    try:
        loaded_model = tf.keras.models.load_model("model_fold3.h5")
        logging.info("✅ Model loaded successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.get_json()
        window = content.get("window")
        if window is None:
            return jsonify({'error': 'Missing "window" key'}), 400
        arr = np.array(window)
        if arr.shape != (10, 5):
            return jsonify({'error': f'Expected shape (10, 5), got {arr.shape}'}), 400
        arr = np.expand_dims(arr, axis=0)
        pred = loaded_model.predict(arr, verbose=0)[0].tolist()
        return jsonify({'prediction': pred})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# ============ Entry Point ============
if __name__ == "__main__":
    mode = input("Ketik 'train' untuk melatih model atau 'run' untuk menjalankan API: ").strip().lower()
    if mode == "train":
        main_training()
    elif mode == "run":
        app.run(host='0.0.0.0', port=5000)
    else:
        print("❌ Pilihan tidak dikenali. Gunakan 'train' atau 'run'.")

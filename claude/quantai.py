# QuantAI: Model dengan TensorFlow untuk prediksi harga (timeseries)
# Developer & AI yang berkontribusi dalam pengembangan kode wajib mengikuti kode etik :
# -Boleh menambahkan komentar inline hanya pada baris kode yang Anda buat, bukan existing!
# -Audit & Analis boleh menambahkan komentar dibaris yang salah atau harus ditingkatkan!
# -Dilarang menghapus existing kode, logika, fungsi yang sudah ada sebelumnya, wajib!
# -Wajib menggunakan bahasa indonesia, jangan gunakan banyak jargon dalam dokumentasi 
# -Wajib menambahkan API untuk menaikan perfoma, wajib tahu logika fungsi yang ditambahkan 
# -Diwajibkan saat editing wajib tahu logika fungsinya, sesudahnya wajib audit sintaks 
# -Diwajibkan melakukan editing end-to-end kode langsung berfungsi bukan potongan kode!
# -Dilarang menghapus header ini, kode etik diterapkan untuk lapisan kemanan data!
# ================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input, Concatenate, Add
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import shutil
import json
import sys
from datetime import datetime
import matplotlib.pyplot as plt

data_file_path = 'XAU_1d_data_processed.json'

try:
    with open(data_file_path, 'r') as f:
        data = json.load(f)

    valid_entries = []
    skipped_rows = 0
    for i in range(len(data) - 1):
        current_row = data[i]
        next_row = data[i+1]

        if i == 0 and isinstance(current_row, list) and len(current_row) > 0 and current_row[0] == "NaT":
            skipped_rows += 1
            continue

        if not isinstance(current_row, list) or len(current_row) < 7:
            skipped_rows += 1
            continue
        if not isinstance(next_row, list) or len(next_row) < 7:
            skipped_rows += 1
            continue

        date_str = current_row[0]
        try:
            date_obj = datetime.fromisoformat(date_str)
            formatted_date = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            skipped_rows += 1
            continue

        is_valid_features = True
        feature_values = []
        for j in range(1, 6):
            if not isinstance(current_row[j], (int, float)):
                is_valid_features = False
                break
            feature_values.append(float(current_row[j]))

        if not is_valid_features:
            skipped_rows += 1
            continue

        is_valid_labels = True
        label_values = []
        for j in [2, 3, 4]: 
            if not isinstance(next_row[j], (int, float)):
                is_valid_labels = False
                break
            label_values.append(float(next_row[j]))

        if not is_valid_labels:
            skipped_rows += 1
            continue
        
        valid_entries.append({'date': formatted_date, 'features': feature_values, 'labels': label_values})

    if not valid_entries:
        raise ValueError("Tidak ada pasangan fitur-label yang valid ditemukan setelah pemrosesan.")

    dates = np.array([item['date'] for item in valid_entries])
    X = np.array([item['features'] for item in valid_entries], dtype=np.float32)
    y = np.array([item['labels'] for item in valid_entries], dtype=np.float32)

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    window_size = 10 

    X_windowed = []
    y_windowed = []
    dates_windowed = []

    for i in range(len(X_scaled) - window_size):
        X_windowed.append(X_scaled[i:i+window_size])
        y_windowed.append(y_scaled[i+window_size])
        dates_windowed.append(dates[i+window_size])

    X_windowed = np.array(X_windowed)
    y_windowed = np.array(y_windowed)
    dates_windowed = np.array(dates_windowed)

    split_idx = int(len(X_windowed) * 0.8)
    X_train, X_test = X_windowed[:split_idx], X_windowed[split_idx:]
    y_train, y_test = y_windowed[:split_idx], y_windowed[split_idx:]
    dates_train, dates_test = dates_windowed[:split_idx], dates_windowed[split_idx:]

    X_original = X
    y_original = y

except FileNotFoundError:
    sys.exit(1)
except json.JSONDecodeError:
    sys.exit(1)
except Exception as e:
    sys.exit(1)

batch_size = 32 
buffer_size = min(1000, len(X_train))

options = tf.data.Options()
options.experimental_deterministic = False
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.with_options(options)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.with_options(options)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

def add_technical_features(x):
    close_prices = x[:, :, 3] 
    sma = tf.reduce_mean(close_prices, axis=1, keepdims=True)
    volatility = tf.math.reduce_std(close_prices, axis=1, keepdims=True)
    last_window_data = x[:, -1, :]
    price_change = (last_window_data[:, 3:4] - x[:, -2, 3:4]) / x[:, -2, 3:4]
    return tf.concat([last_window_data, sma, volatility, price_change], axis=1)

def build_lstm_attention_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    attention_output = MultiHeadAttention(
        num_heads=2, key_dim=32, dropout=0.1
    )(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(output_dim, activation=None)(x) 
    return Model(inputs=inputs, outputs=outputs)

def build_cnn_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_dim, activation=None)(x)
    return Model(inputs=inputs, outputs=outputs)

def build_hybrid_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(50)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_dim, activation=None)(x)
    return Model(inputs=inputs, outputs=outputs)

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch'
)

def cosine_decay_schedule(epoch, lr):
    initial_lr = 0.001
    min_lr = 0.0001
    decay_epochs = 40
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        progress = min(1.0, (epoch - warmup_epochs) / decay_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay_schedule)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_quantai_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

tscv = TimeSeriesSplit(n_splits=5)
models_lstm_attention = []
models_cnn = []
models_hybrid = []
histories = []

fold = 0
for train_idx, val_idx in tscv.split(X_train):
    fold += 1
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    fold_train_dataset = tf.data.Dataset.from_tensor_slices((X_fold_train, y_fold_train))
    fold_train_dataset = fold_train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    fold_val_dataset = tf.data.Dataset.from_tensor_slices((X_fold_val, y_fold_val))
    fold_val_dataset = fold_val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    model_lstm_attention = build_lstm_attention_model(
        input_shape=(window_size, X_train.shape[2]), 
        output_dim=y_train.shape[1]
    )
    model_lstm_attention.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    history = model_lstm_attention.fit(
        fold_train_dataset,
        epochs=30, 
        validation_data=fold_val_dataset,
        callbacks=[early_stopping, lr_scheduler, tensorboard_callback],
        verbose=1
    )
    histories.append(history)
    models_lstm_attention.append(model_lstm_attention)
    
    model_cnn = build_cnn_model(
        input_shape=(window_size, X_train.shape[2]), 
        output_dim=y_train.shape[1]
    )
    model_cnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    model_cnn.fit(
        fold_train_dataset,
        epochs=30,
        validation_data=fold_val_dataset,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    models_cnn.append(model_cnn)
    
    model_hybrid = build_hybrid_model(
        input_shape=(window_size, X_train.shape[2]), 
        output_dim=y_train.shape[1]
    )
    model_hybrid.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    model_hybrid.fit(
        fold_train_dataset,
        epochs=30,
        validation_data=fold_val_dataset,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    models_hybrid.append(model_hybrid)

def ensemble_predict(models_list, x_data):
    predictions = []
    for model in models_list:
        pred = model.predict(x_data, verbose=0)
        predictions.append(pred)
    return np.mean(predictions, axis=0)

lstm_attention_preds = ensemble_predict(models_lstm_attention, X_test)
cnn_preds = ensemble_predict(models_cnn, X_test)
hybrid_preds = ensemble_predict(models_hybrid, X_test)

ensemble_preds = (lstm_attention_preds + cnn_preds + hybrid_preds) / 3

ensemble_preds_original = scaler_y.inverse_transform(ensemble_preds)
y_test_original = scaler_y.inverse_transform(y_test)

ensemble_mae = np.mean(np.abs(ensemble_preds_original - y_test_original))
ensemble_mse = np.mean((ensemble_preds_original - y_test_original)**2)
ensemble_rmse = np.sqrt(ensemble_mse)

for i, name in enumerate(['High', 'Low', 'Close']):
    mae = np.mean(np.abs(ensemble_preds_original[:, i] - y_test_original[:, i]))
    mape = np.mean(np.abs((y_test_original[:, i] - ensemble_preds_original[:, i]) / y_test_original[:, i])) * 100
    # print(f"{name} - MAE: {mae:.4f}, MAPE: {mape:.2f}%") # Removed as per instruction

num_samples_to_show = 20
last_indices = np.arange(len(y_test_original) - num_samples_to_show, len(y_test_original))

plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
plt.plot(last_indices, y_test_original[last_indices, 0], 'go-', label='Aktual High')
plt.plot(last_indices, ensemble_preds_original[last_indices, 0], 'ro-', label='Prediksi High')
plt.title('Perbandingan Harga High - Aktual vs Prediksi')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(last_indices, y_test_original[last_indices, 1], 'go-', label='Aktual Low')
plt.plot(last_indices, ensemble_preds_original[last_indices, 1], 'ro-', label='Prediksi Low')
plt.title('Perbandingan Harga Low - Aktual vs Prediksi')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(last_indices, y_test_original[last_indices, 2], 'go-', label='Aktual Close')
plt.plot(last_indices, ensemble_preds_original[last_indices, 2], 'ro-', label='Prediksi Close')
plt.title('Perbandingan Harga Close - Aktual vs Prediksi')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('quantai_ensemble_predictions.png')

model_artifacts_dir = 'quantai_model_artifacts'
os.makedirs(model_artifacts_dir, exist_ok=True)

best_model = models_lstm_attention[0] 
best_model.save(f'{model_artifacts_dir}/quantai_best_model.h5')

np.save(f'{model_artifacts_dir}/x_min.npy', scaler_X.data_min_)
np.save(f'{model_artifacts_dir}/x_scale.npy', scaler_X.scale_)
np.save(f'{model_artifacts_dir}/y_min.npy', scaler_y.data_min_)
np.save(f'{model_artifacts_dir}/y_scale.npy', scaler_y.scale_)

model_info = {
    'feature_names': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'output_names': ['Next_High', 'Next_Low', 'Next_Close'],
    'window_size': window_size,
    'batch_size': batch_size,
    'version': '2.0.0'
}
with open(f'{model_artifacts_dir}/model_info.json', 'w') as f:
    json.dump(model_info, f)

def predict_next_day_prices(model, latest_data, scaler_X, scaler_y, window_size=10):
    if len(latest_data) < window_size:
        raise ValueError(f"Data tidak cukup. Butuh minimal {window_size} hari data.")
    
    recent_data = latest_data[-window_size:]
    recent_data_scaled = scaler_X.transform(recent_data)
    model_input = recent_data_scaled.reshape(1, window_size, -1)
    prediction_scaled = model.predict(model_input, verbose=0)
    prediction_original = scaler_y.inverse_transform(prediction_scaled)
    
    return {
        'high': float(prediction_original[0, 0]),
        'low': float(prediction_original[0, 1]),
        'close': float(prediction_original[0, 2])
    }

test_input = X_original[-window_size:] 
predicted_prices = predict_next_day_prices(best_model, test_input, scaler_X, scaler_y, window_size)
# print(f"Prediksi harga emas untuk hari berikutnya: {predicted_prices}") # Removed as per instruction

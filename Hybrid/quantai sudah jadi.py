# QuantAI: Model Prediksi Harga XAU dengan TensorFlow Data API (Ditingkatkan)
# Developer & AI yang berkontribusi dalam pengembangan kode wajib mengikuti kode etik :
# -Rilarang menghapus existing kode, logika, fungsi yang sudah ada 
# -Wajib menggunakan bahasa indonesia, jangan gunakan banyak jargon dalam dokumentasi 
# -Wajib menambahkan API untuk menaikan perfoma, wajib tahu logika fungsi yang ditambahkan 
# -Diwajibkan sebelum melakukan editinf sudah tahu apa yang akan ditambahkan, sesudah editing wajib audit sintaks 
# -Diwajibkan melakukan editing end-to-end kode langsung berfungsi bukan potongan kode!
# -Dilarang menghapus header ini, kode etik diterapkan untuk lapisan kemanan data!
# ================================================================

# Langkah 1: Impor Library yang dibutuhkan
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

print("Langkah 1: Library yang dibutuhkan berhasil diimpor.")

# Path ke file data yang diunggah
data_file_path = 'XAU_1d_data_processed.json'

# Langkah 2: Persiapan Data (Untuk Prediksi Harga dengan Tanggal)
print("\nLangkah 2: Memuat dan memproses data real dari", data_file_path, "untuk prediksi harga...")

# Memuat data dari file JSON dan memprosesnya
try:
    with open(data_file_path, 'r') as f:
        data = json.load(f)
    print("Data berhasil dimuat dari", data_file_path)

    valid_entries = []
    skipped_rows = 0
    for i in range(len(data) - 1):
        current_row = data[i]
        next_row = data[i+1]

        if i == 0 and isinstance(current_row, list) and len(current_row) > 0 and current_row[0] == "NaT":
            print("Melewati baris header pertama.")
            skipped_rows += 1
            continue

        if not isinstance(current_row, list) or len(current_row) < 7:
            skipped_rows += 1
            continue
        if not isinstance(next_row, list) or len(next_row) < 7:
             print(f"Melewati baris {i+1} karena baris berikutnya ({i+2}) tidak valid atau jumlah kolom kurang: {next_row}")
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

    print(f"Melewati {skipped_rows} baris yang tidak valid, merupakan header, atau tidak memiliki data label hari berikutnya.")

    if not valid_entries:
        raise ValueError("Tidak ada pasangan fitur-label yang valid ditemukan setelah pemrosesan.")

    dates = np.array([item['date'] for item in valid_entries])
    X = np.array([item['features'] for item in valid_entries], dtype=np.float32)
    y = np.array([item['labels'] for item in valid_entries], dtype=np.float32)

    print("Data real berhasil diproses dan dikonversi.")
    print("Shape Tanggal:", dates.shape)
    print("Shape Fitur (X):", X.shape)
    print("Shape Label (y):", y.shape)

    # Penskalaan Fitur dan Label
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    print("Fitur data real berhasil diskalakan.")

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)
    print("Label data real berhasil diskalakan.")

    # Pengaturan window size untuk time series
    window_size = 10  # Menggunakan 10 hari terakhir untuk prediksi

    # Membuat dataset berbasis sliding window untuk time series
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

    print("Dataset windowed berhasil dibuat:")
    print("X_windowed shape:", X_windowed.shape)
    print("y_windowed shape:", y_windowed.shape)
    print("dates_windowed shape:", dates_windowed.shape)

    # Membagi data yang sudah diproses menjadi set pelatihan dan pengujian
    # Menggunakan pemisahan kronologis untuk data time series (80% awal untuk training, 20% akhir untuk testing)
    split_idx = int(len(X_windowed) * 0.8)
    X_train, X_test = X_windowed[:split_idx], X_windowed[split_idx:]
    y_train, y_test = y_windowed[:split_idx], y_windowed[split_idx:]
    dates_train, dates_test = dates_windowed[:split_idx], dates_windowed[split_idx:]

    print("Data time series dibagi secara kronologis:")
    print("X_train={}, X_test={}".format(X_train.shape, X_test.shape))
    print("y_train={}, y_test={}".format(y_train.shape, y_test.shape))
    print("dates_train={}, dates_test={}".format(dates_train.shape, dates_test.shape))

    # Simpan data asli untuk analisis akhir
    X_original = X
    y_original = y

except FileNotFoundError:
    print("ERROR FATAL: File data", data_file_path, "tidak ditemukan.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"ERROR FATAL: Gagal mem-parse file JSON '{data_file_path}'. Pastikan format file JSON valid.")
    sys.exit(1)
except Exception as e:
    print("ERROR FATAL saat memuat atau memproses data:", e)
    sys.exit(1)

# Langkah 2b: Implementasi TensorFlow Data API untuk performa yang lebih baik
print("\nLangkah 2b: Mengimplementasikan TensorFlow Data API...")

# Definisikan parameter untuk dataset
batch_size = 32  # Batch size yang lebih kecil untuk mencegah overfitting
buffer_size = min(1000, len(X_train))

# Konfigurasi tf.data untuk performa yang dioptimalkan
options = tf.data.Options()
options.experimental_deterministic = False
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True

# Membuat dataset training dengan tf.data API
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.with_options(options)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# Membuat dataset testing dengan tf.data API
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.with_options(options)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print("TensorFlow Data API berhasil diimplementasikan dengan optimasi performa.")

# Langkah 3: Persiapan untuk Feature Engineering dan penanganan indikator teknikal
print("\nLangkah 3: Mengimplementasikan feature engineering dalam pipeline...")

def add_technical_features(x):
    """Lapisan kustom untuk menghitung indikator teknikal pada data window"""
    # x memiliki bentuk [batch, window, features]
    
    # Ekstrak harga close dari semua data window
    close_prices = x[:, :, 3]
    
    # Hitung SMA (Simple Moving Average)
    sma = tf.reduce_mean(close_prices, axis=1, keepdims=True)
    
    # Hitung volatilitas (standar deviasi)
    volatility = tf.math.reduce_std(close_prices, axis=1, keepdims=True)
    
    # Ambil data terbaru dari window
    last_window_data = x[:, -1, :]
    
    # Hitung perubahan harga relatif terhadap hari sebelumnya
    price_change = (last_window_data[:, 3:4] - x[:, -2, 3:4]) / x[:, -2, 3:4]
    
    # Gabungkan fitur-fitur teknikal
    return tf.concat([last_window_data, sma, volatility, price_change], axis=1)

# Langkah 4: Definisi Arsitektur Model Hybrid (LSTM + Attention)
print("\nLangkah 4: Mendefinisikan arsitektur model hybrid untuk prediksi harga multi-output...")

def build_lstm_attention_model(input_shape, output_dim):
    """Arsitektur model hybrid dengan LSTM dan Attention untuk data time series"""
    inputs = Input(shape=input_shape)
    
    # Bagian LSTM untuk menangkap pola temporal
    x = LSTM(64, return_sequences=True)(inputs)
    
    # Multi-Head Attention untuk menangkap hubungan antar waktu
    attention_output = MultiHeadAttention(
        num_heads=2, key_dim=32, dropout=0.1
    )(x, x)
    
    # Residual connection dan normalisasi
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    
    # Global pooling untuk mengekstrak fitur penting
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers dengan dropout untuk mencegah overfitting
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    
    # Output layer untuk high, low, close
    outputs = Dense(output_dim, activation=None)(x)
    
    return Model(inputs=inputs, outputs=outputs)

# Langkah 5: Membangun model ensemble (rata-rata dari 3 model berbeda)
print("\nLangkah 5: Membangun model ensemble...")

def build_cnn_model(input_shape, output_dim):
    """Model CNN untuk time series"""
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
    """Model hybrid dengan CNN + LSTM"""
    inputs = Input(shape=input_shape)
    
    # Bagian CNN
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Bagian LSTM
    x = LSTM(50)(x)
    
    # Dense layers
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_dim, activation=None)(x)
    
    return Model(inputs=inputs, outputs=outputs)

# Langkah 6: Persiapan training dengan cross-validation dan learning rate scheduling
print("\nLangkah 6: Menyiapkan cross-validation dan learning rate scheduling...")

# Setup TensorBoard
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch'
)

# Learning rate scheduling yang lebih canggih
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

# Early stopping dengan monitoring loss
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint untuk menyimpan model terbaik
model_checkpoint = ModelCheckpoint(
    'best_quantai_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Langkah 7: Implementasi training dengan cross-validation untuk ensemble
print("\nLangkah 7: Melatih model ensemble dengan cross-validation...")

# Menggunakan TimeSeriesSplit untuk time series data
tscv = TimeSeriesSplit(n_splits=5)

# Penyimpanan model-model hasil training
models_lstm_attention = []
models_cnn = []
models_hybrid = []

# Metrics untuk evaluasi
histories = []

# Cross-validation untuk model LSTM-Attention
fold = 0
for train_idx, val_idx in tscv.split(X_train):
    fold += 1
    print(f"\nMelatih fold {fold} dari 5 untuk model LSTM-Attention...")
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    # Membuat dataset untuk fold ini
    fold_train_dataset = tf.data.Dataset.from_tensor_slices((X_fold_train, y_fold_train))
    fold_train_dataset = fold_train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    fold_val_dataset = tf.data.Dataset.from_tensor_slices((X_fold_val, y_fold_val))
    fold_val_dataset = fold_val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Membangun dan kompilasi model
    model_lstm_attention = build_lstm_attention_model(
        input_shape=(window_size, X_train.shape[2]), 
        output_dim=y_train.shape[1]
    )
    
    model_lstm_attention.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    # Melatih model
    history = model_lstm_attention.fit(
        fold_train_dataset,
        epochs=30,  # Lebih sedikit epoch untuk mencegah overfitting
        validation_data=fold_val_dataset,
        callbacks=[early_stopping, lr_scheduler, tensorboard_callback],
        verbose=1
    )
    
    histories.append(history)
    models_lstm_attention.append(model_lstm_attention)
    
    # Kompilasi dan training model CNN
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
    
    # Kompilasi dan training model hybrid
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

print("Pelatihan cross-validation untuk model ensemble selesai.")

# Langkah 8: Evaluasi Model ensemble pada data pengujian
print("\nLangkah 8: Mengevaluasi model ensemble pada data pengujian...")

# Fungsi untuk melakukan prediksi ensemble
def ensemble_predict(models_list, x_data):
    predictions = []
    for model in models_list:
        pred = model.predict(x_data, verbose=0)
        predictions.append(pred)
    
    # Rata-rata prediksi dari semua model
    return np.mean(predictions, axis=0)

# Prediksi dari setiap jenis model
lstm_attention_preds = ensemble_predict(models_lstm_attention, X_test)
cnn_preds = ensemble_predict(models_cnn, X_test)
hybrid_preds = ensemble_predict(models_hybrid, X_test)

# Ensemble final (rata-rata dari semua model)
ensemble_preds = (lstm_attention_preds + cnn_preds + hybrid_preds) / 3

# Inverse transform untuk mendapatkan nilai asli
ensemble_preds_original = scaler_y.inverse_transform(ensemble_preds)
y_test_original = scaler_y.inverse_transform(y_test)

# Menghitung metrik untuk ensemble
ensemble_mae = np.mean(np.abs(ensemble_preds_original - y_test_original))
ensemble_mse = np.mean((ensemble_preds_original - y_test_original)**2)
ensemble_rmse = np.sqrt(ensemble_mse)

print("Metrik Ensemble Model:")
print(f"MAE: {ensemble_mae:.4f}")
print(f"MSE: {ensemble_mse:.4f}")
print(f"RMSE: {ensemble_rmse:.4f}")

# Menghitung metrik per komponen harga (High, Low, Close)
for i, name in enumerate(['High', 'Low', 'Close']):
    mae = np.mean(np.abs(ensemble_preds_original[:, i] - y_test_original[:, i]))
    mape = np.mean(np.abs((y_test_original[:, i] - ensemble_preds_original[:, i]) / y_test_original[:, i])) * 100
    print(f"{name} - MAE: {mae:.4f}, MAPE: {mape:.2f}%")

# Langkah 9: Visualisasi performa model ensemble
print("\nLangkah 9: Membuat visualisasi performa model...")

# Mengambil 20 sampel terakhir untuk visualisasi yang lebih jelas
num_samples_to_show = 20
last_indices = np.arange(len(y_test_original) - num_samples_to_show, len(y_test_original))

plt.figure(figsize=(15, 12))

# Plot High
plt.subplot(3, 1, 1)
plt.plot(last_indices, y_test_original[last_indices, 0], 'go-', label='Aktual High')
plt.plot(last_indices, ensemble_preds_original[last_indices, 0], 'ro-', label='Prediksi High')
plt.title('Perbandingan Harga High - Aktual vs Prediksi')
plt.legend()
plt.grid(True)

# Plot Low
plt.subplot(3, 1, 2)
plt.plot(last_indices, y_test_original[last_indices, 1], 'go-', label='Aktual Low')
plt.plot(last_indices, ensemble_preds_original[last_indices, 1], 'ro-', label='Prediksi Low')
plt.title('Perbandingan Harga Low - Aktual vs Prediksi')
plt.legend()
plt.grid(True)

# Plot Close
plt.subplot(3, 1, 3)
plt.plot(last_indices, y_test_original[last_indices, 2], 'go-', label='Aktual Close')
plt.plot(last_indices, ensemble_preds_original[last_indices, 2], 'ro-', label='Prediksi Close')
plt.title('Perbandingan Harga Close - Aktual vs Prediksi')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('quantai_ensemble_predictions.png')
print("Grafik perbandingan prediksi ensemble disimpan sebagai 'quantai_ensemble_predictions.png'")

# Langkah 10: Menyimpan model terbaik dan komponen pendukungnya
print("\nLangkah 10: Menyimpan model terbaik dan komponen pendukungnya...")

# Buat direktori untuk model artifacts
model_artifacts_dir = 'quantai_model_artifacts'
os.makedirs(model_artifacts_dir, exist_ok=True)

# Ambil model terbaik (model pertama dari LSTM-Attention)
best_model = models_lstm_attention[0]

# Menyimpan dalam format HDF5
best_model.save(f'{model_artifacts_dir}/quantai_best_model.h5')

# Menyimpan metadata scaler
np.save(f'{model_artifacts_dir}/x_min.npy', scaler_X.data_min_)
np.save(f'{model_artifacts_dir}/x_scale.npy', scaler_X.scale_)
np.save(f'{model_artifacts_dir}/y_min.npy', scaler_y.data_min_)
np.save(f'{model_artifacts_dir}/y_scale.npy', scaler_y.scale_)

# Menyimpan info tambahan
model_info = {
    'feature_names': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'output_names': ['Next_High', 'Next_Low', 'Next_Close'],
    'window_size': window_size,
    'batch_size': batch_size,
    'version': '2.0.0'
}

with open(f'{model_artifacts_dir}/model_info.json', 'w') as f:
    json.dump(model_info, f)

print(f"Model dan komponen pendukung berhasil disimpan di direktori '{model_artifacts_dir}'")

# Langkah 11: Fungsi untuk prediksi real-time dengan model terbaik
print("\nLangkah 11: Mempersiapkan fungsi prediksi untuk penggunaan real-time...")

def predict_next_day_prices(model, latest_data, scaler_X, scaler_y, window_size=10):
    """
    Fungsi untuk memprediksi harga emas di hari berikutnya
    
    Args:
        model: Model TensorFlow yang telah dilatih
        latest_data: Data OHLCV terkini (minimal window_size hari)
        scaler_X: Scaler untuk data input
        scaler_y: Scaler untuk data output
        window_size: Ukuran jendela yang digunakan
    
    Returns:
        dict: Prediksi harga High, Low, Close untuk hari berikutnya
    """
    # Pastikan data cukup untuk window
    if len(latest_data) < window_size:
        raise ValueError(f"Data tidak cukup. Butuh minimal {window_size} hari data.")
    
    # Ambil window_size data terbaru
    recent_data = latest_data[-window_size:]
    
    # Skalakan data
    recent_data_scaled = scaler_X.transform(recent_data)
    
    # Reshape untuk input model [1, window_size, features]
    model_input = recent_data_scaled.reshape(1, window_size, -1)
    
    # Prediksi
    prediction_scaled = model.predict(model_input, verbose=0)
    
    # Inverse transform ke nilai asli
    prediction_original = scaler_y.inverse_transform(prediction_scaled)
    
    # Format hasil
    return {
        'high': float(prediction_original[0, 0]),
        'low': float(prediction_original[0, 1]),
        'close': float(prediction_original[0, 2])
    }

# Contoh penggunaan fungsi prediksi
print("Contoh penggunaan fungsi prediksi:")
test_input = X_original[-window_size:]
predicted_prices = predict_next_day_prices(best_model, test_input, scaler_X, scaler_y, window_size)
print(f"Prediksi harga emas untuk hari berikutnya: {predicted_prices}")

# Langkah 12: Ringkasan akhir
print("\nQuantAI: Pipeline AI telah berhasil dieksekusi dengan peningkatan:")
print("1. Arsitektur model berbasis LSTM + Attention untuk menangkap pola temporal")
print("2. Ensemble dari tiga arsitektur model berbeda untuk meningkatkan keandalan")
print("3. Pendekatan sliding window untuk memperkaya konteks prediksi time series")
print("4. Learning rate scheduling dan early stopping untuk mencegah overfitting")
print("5. Cross-validation khusus time series untuk evaluasi yang lebih baik")
print("6. Penyimpanan model dan komponen untuk deployment yang mudah")
print("7. Fungsi prediksi siap pakai untuk implementasi real-time")

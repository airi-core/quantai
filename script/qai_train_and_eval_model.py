import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam # Menggunakan Adam standar Keras
import numpy as np
# Pastikan variabel X_train, y_train, X_val, y_val, X_test, y_test, dan scaler_target
# sudah tersedia dari script persiapan data sebelumnya.

# --- Konfigurasi Model (sesuai YAML, disesuaikan dengan window_size 256) ---
MODEL_NAME = 'quantAI'
WINDOW_SIZE = 256 # Harus sama dengan window_size yang digunakan saat persiapan data
INPUT_FEATURES_COUNT = 5 # OHLCV
OUTPUT_FEATURES_COUNT = 3 # HLC

# Perhatikan: Konfigurasi optimizer di YAML mencantumkan 'momentum: 0.9' untuk Adam.
# Optimizer Adam standar di Keras/TensorFlow tidak memiliki parameter 'momentum' dalam bentuk ini.
# Ini memiliki beta_1 (default 0.9) dan beta_2 (default 0.999).
# Kita akan menggunakan Adam standar Keras dengan lr=0.01 dan beta_1 default (0.9).
# Learning rate 0.01 cukup tinggi, mungkin perlu diturunkan atau menggunakan learning rate schedule.
LEARNING_RATE = 0.01

# --- Langkah 8: Bangun Model Keras ---
print(f"Membangun model '{MODEL_NAME}'...")

model = Sequential([
    # Layer LSTM pertama
    LSTM(units=128,
         return_sequences=True, # Penting: return_sequences=True agar outputnya sekuens untuk layer berikutnya
         activation='tanh',
         recurrent_activation='sigmoid',
         recurrent_dropout=0.0,
         kernel_regularizer=l2(0.0001),
         input_shape=(WINDOW_SIZE, INPUT_FEATURES_COUNT)), # Input shape sesuai data
    BatchNormalization(),
    Dropout(rate=0.236),

    # Layer Bidirectional LSTM
    Bidirectional(
        LSTM(units=64,
             return_sequences=False, # return_sequences=False karena ini layer LSTM terakhir sebelum Dense
             activation='tanh',
             recurrent_activation='sigmoid')
        ),
    BatchNormalization(),
    Dropout(rate=0.382),

    # Layer Dense pertama
    Dense(units=32,
          activation='relu',
          kernel_regularizer=l2(0.0001)),

    # Layer Dense output
    Dense(units=OUTPUT_FEATURES_COUNT, # Output 3 unit untuk HLC
          activation='linear') # Aktivasi linear untuk tugas regresi
])

# --- Langkah 9: Compile Model ---
print("Mengompilasi model...")
# Menggunakan optimizer Adam dengan learning rate yang ditentukan
optimizer = Adam(learning_rate=LEARNING_RATE)

# Menggunakan Mean Squared Error (MSE) sebagai loss function untuk regresi
# Menambahkan Mean Absolute Error (MAE) sebagai metrik tambahan untuk pemantauan
model.compile(optimizer=optimizer,
              loss='mse', # Mean Squared Error
              metrics=['mae', 'mse']) # Mean Absolute Error dan Mean Squared Error

model.summary()

# --- Langkah 10: Latih Model ---
print("\nMemulai pelatihan model...")
# Tentukan jumlah epoch dan batch size
EPOCHS = 50 # Anda bisa menyesuaikan jumlah epoch ini
BATCH_SIZE = 32 # Anda bisa menyesuaikan batch size ini

# Melatih model menggunakan data training dan memvalidasinya dengan data validation
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    verbose=1) # verbose=1 untuk menampilkan progress bar

print("\nPelatihan selesai.")

# --- Langkah 11: Evaluasi Model ---
print("\nMengevaluasi model pada data pengujian (scaled)...")
loss_scaled, mae_scaled, mse_scaled = model.evaluate(X_test, y_test, verbose=0)

print(f"Hasil Evaluasi pada Data Pengujian (Scaled):")
print(f"  Loss (MSE): {loss_scaled:.6f}")
print(f"  MAE: {mae_scaled:.6f}")
print(f"  MSE: {mse_scaled:.6f}")

# --- Evaluasi pada skala harga asli (Inverse Scaling) ---
print("\nMengevaluasi model pada skala harga asli (Inverse Scaling)...")

# Lakukan prediksi pada data pengujian
y_pred_scaled = model.predict(X_test)

# Lakukan inverse scaling pada prediksi dan nilai y_test asli
# Gunakan scaler_target yang sudah dilatih pada data HLC mentah
y_test_original_scale = scaler_target.inverse_transform(y_test)
y_pred_original_scale = scaler_target.inverse_transform(y_pred_scaled)

# Hitung metrik (MAE dan MSE) pada skala harga asli
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_original_scale = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
mse_original_scale = mean_squared_error(y_test_original_scale, y_pred_original_scale)
rmse_original_scale = np.sqrt(mse_original_scale) # RMSE seringkali lebih mudah diinterpretasikan

print(f"Hasil Evaluasi pada Data Pengujian (Skala Harga Asli):")
print(f"  MAE: {mae_original_scale:.4f}")
print(f"  MSE: {mse_original_scale:.4f}")
print(f"  RMSE: {rmse_original_scale:.4f}") # Root Mean Squared Error

# --- Opsional: Tampilkan plot loss dan metrik selama pelatihan ---
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt # Untuk visualisasi riwayat pelatihan
import os # Untuk membuat direktori

# --- Pastikan variabel berikut tersedia dari script sebelumnya ---
# model: Model Keras yang sudah dilatih
# history: Objek History dari proses model.fit()
# X_test: Data input pengujian (sudah discale dan berbentuk sekuens)
# y_test: Data target pengujian (sudah discale)
# scaler_target: Scaler yang digunakan untuk menormalisasi target HLC (penting untuk inverse scaling)
# scaler_input: Scaler untuk input OHLCV (mungkin perlu untuk inverse scaling input jika diperlukan)

# --- Konfigurasi ---
MODEL_SAVE_PATH = 'saved_models/quantAI_lstm_ohlcv_hlc_w256' # Path untuk menyimpan model
# Pastikan nama folder 'saved_models' sudah ada atau akan dibuat

# --- Langkah Setelah Pelatihan Selesai ---

print("\n--- Proses Pasca-Pelatihan ---")

# --- 1. Menyimpan Model yang Sudah Dilatih (Craft -> Clone) ---
# Menyimpan model sangat penting agar bisa digunakan nanti tanpa melatih ulang
# Kita akan simpan dalam format SavedModel TensorFlow
print(f"Menyimpan model ke: {MODEL_SAVE_PATH}")
try:
    # Buat direktori jika belum ada
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print("Model berhasil disimpan.")
except Exception as e:
    print(f"ERROR saat menyimpan model: {e}")

# Untuk memuat kembali model nanti:
# loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
# print("Model berhasil dimuat kembali.")


# --- 2. Menganalisis Riwayat Pelatihan (Logic-building) ---
# Melihat plot loss dan metrik selama pelatihan bisa memberikan insight
# tentang konvergensi model dan potensi overfitting.

print("\nMenampilkan riwayat pelatihan...")

# Ambil data loss dan metrik dari objek history
loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']
mse = history.history['mse']
val_mse = history.history['val_mse']

epochs = range(1, len(loss) + 1)

# Plot Loss (MSE)
plt.figure(figsize=(12, 6))
plt.plot(epochs, loss, 'r', label='Training Loss (MSE)')
plt.plot(epochs, val_loss, 'b', label='Validation Loss (MSE)')
plt.title('Training and Validation Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Plot MAE
plt.figure(figsize=(12, 6))
plt.plot(epochs, mae, 'r', label='Training MAE')
plt.plot(epochs, val_mae, 'b', label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.show()

# Catatan: Jika ada gap besar antara training dan validation metrik,
# ini bisa jadi tanda overfitting. Jika keduanya datar atau naik,
# bisa jadi underfitting atau learning rate terlalu tinggi/rendah.


# --- 3. Membuat Prediksi Menggunakan Model yang Sudah Dilatih (Craft) ---
# Mendemonstrasikan cara menggunakan model untuk inferensi

print("\nMembuat prediksi pada data pengujian (scaled)...")
y_pred_scaled = model.predict(X_test)
print(f"Bentuk hasil prediksi (scaled): {y_pred_scaled.shape}")

# --- 4. Inverse Scaling Hasil Prediksi (Logic-building) ---
# Mengembalikan hasil prediksi ke skala harga asli menggunakan scaler_target

print("Melakukan inverse scaling pada prediksi dan target asli...")
try:
    y_test_original_scale = scaler_target.inverse_transform(y_test)
    y_pred_original_scale = scaler_target.inverse_transform(y_pred_scaled)

    print("Inverse scaling berhasil.")
    print("\nCuplikan 5 hasil prediksi pertama (Skala Harga Asli):")
    # Tampilkan beberapa contoh prediksi vs nilai asli
    for i in range(5):
        print(f"  Sampel {i+1}:")
        print(f"    Asli (HLC): {y_test_original_scale[i]}")
        print(f"    Prediksi (HLC): {y_pred_original_scale[i]}")
        print("-" * 20)

    # Anda bisa menghitung metrik lagi di sini pada skala asli jika perlu,
    # seperti yang sudah dilakukan di script sebelumnya.
    # from sklearn.metrics import mean_absolute_error, mean_squared_error
    # mae_final = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
    # mse_final = mean_squared_error(y_test_original_scale, y_pred_original_scale)
    # rmse_final = np.sqrt(mse_final)
    # print(f"\nFinal MAE (Original Scale): {mae_final:.4f}")
    # print(f"Final RMSE (Original Scale): {rmse_final:.4f}")


except Exception as e:
    print(f"ERROR saat melakukan inverse scaling: {e}")
    print("Pastikan variabel 'scaler_target' tersedia dan benar.")


# --- 5. Pentingnya Arsitektur Directory Proyek (Architecture) ---
# Untuk mengelola proyek AI dengan baik, struktur folder yang rapi itu krusial.
# Ini membantu dalam:
# - Organisasi: Menemukan file dengan mudah.
# - Reproduksibilitas: Memastikan orang lain (atau Anda di masa depan) bisa menjalankan kode.
# - Kolaborasi: Mempermudah kerja tim.
# - Versioning: Melacak perubahan pada kode, data, dan model.

print("\n--- Pentingnya Arsitektur Directory Proyek ---")
print("Untuk proyek AI yang sustainable, sangat disarankan menata file Anda dalam struktur yang logis.")
print("Contoh struktur sederhana:")
print("my_quantai_project/")
print("├── data/")         # Data mentah dan data yang sudah diproses
print("│   └── raw/")
print("│   └── processed/") # XAU_15m_data_processed.json bisa di sini
print("├── notebooks/")    # Notebook eksplorasi atau eksperimen (seperti Colab ini)
print("├── scripts/")      # Script Python untuk persiapan data, pelatihan, evaluasi, inferensi
print("│   └── data_prep.py")
print("│   └── train_eval.py")
print("│   └── predict.py")
print("├── models/")       # Model yang sudah dilatih (folder 'saved_models' bisa di sini)
print("├── config/")       # File konfigurasi (seperti model_config.yaml Anda)
print("├── results/")      # Hasil eksperimen, plot, metrik
print("├── .gitignore")    # Untuk Git (mengabaikan file besar atau sensitif)
print("└── README.md")     # Penjelasan proyek

print("\nMenata proyek seperti ini akan sangat membantu pengelolaan dan pengembangan 'quantAI' Anda ke depan.")


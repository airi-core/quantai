import tensorflow as tf
import numpy as np
import joblib # Library untuk memuat scaler
import pandas as pd # Mungkin diperlukan untuk memproses data baru
import os # Untuk mengelola path file

# --- Konfigurasi ---
# Sesuaikan path ini agar sesuai dengan lokasi di repository GitHub Anda
# Jika Anda menjalankan script ini dari root folder 'my_quantai_project'
MODEL_PATH = 'models/saved_models/quantAI_lstm_ohlcv_hlc_w256' # Path ke model yang disimpan
SCALER_INPUT_PATH = 'data/processed/scalers/scaler_input.pkl' # Path ke scaler input
SCALER_TARGET_PATH = 'data/processed/scalers/scaler_target.pkl' # Path ke scaler target

WINDOW_SIZE = 256 # Harus sama dengan window size saat pelatihan
INPUT_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume'] # Fitur input
TARGET_FEATURES = ['High', 'Low', 'Close'] # Fitur target (untuk inverse scaling)

# --- Fungsi Bantu untuk Memuat Model dan Scaler ---
def load_model_and_scalers(model_path, scaler_input_path, scaler_target_path):
    """Memuat model Keras yang sudah dilatih dan objek scaler."""
    try:
        print(f"Memuat model dari: {model_path}")
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model berhasil dimuat.")

        print(f"Memuat scaler input dari: {scaler_input_path}")
        loaded_scaler_input = joblib.load(scaler_input_path)
        print("Scaler input berhasil dimuat.")

        print(f"Memuat scaler target dari: {scaler_target_path}")
        loaded_scaler_target = joblib.load(scaler_target_path)
        print("Scaler target berhasil dimuat.")

        return loaded_model, loaded_scaler_input, loaded_scaler_target

    except FileNotFoundError as e:
        print(f"ERROR: File tidak ditemukan - {e}. Pastikan path file model dan scaler sudah benar.")
        return None, None, None
    except Exception as e:
        print(f"ERROR saat memuat model atau scaler: {e}")
        return None, None, None

# --- Fungsi Bantu untuk Memproses Data Baru ---
def preprocess_new_data(new_data_df, scaler_input, window_size, input_features):
    """
    Memproses data baru (DataFrame) menjadi format input yang siap untuk model.
    Asumsi new_data_df berisi data OHLCV terbaru yang cukup untuk satu window.
    """
    # Pastikan DataFrame memiliki kolom input yang dibutuhkan
    if not all(feature in new_data_df.columns for feature in input_features):
        print("ERROR: DataFrame data baru tidak memiliki kolom fitur input yang lengkap.")
        return None

    # Ambil nilai fitur input
    data_input_raw = new_data_df[input_features].values

    # Scale data input baru menggunakan scaler yang sudah dilatih
    # Perhatikan: scaler.transform() butuh input 2D array (samples, features)
    data_input_scaled = scaler_input.transform(data_input_raw)

    # Bentuk data menjadi window sekuensial
    # Untuk prediksi real-time, Anda mungkin hanya punya 1 window terbaru
    # Asumsi data_input_scaled sudah memiliki panjang minimal window_size
    if len(data_input_scaled) < window_size:
        print(f"ERROR: Panjang data baru ({len(data_input_scaled)}) kurang dari window size ({window_size}). Tidak bisa membuat window.")
        return None

    # Ambil window TERAKHIR dari data baru sebagai input untuk prediksi
    input_window_scaled = data_input_scaled[-window_size:] # Ambil 256 titik data terakhir

    # Reshape untuk input model: (jumlah_sampel, window_size, jumlah_fitur)
    # Karena hanya 1 window, jumlah_sampel = 1
    input_for_prediction = np.reshape(input_window_scaled, (1, window_size, len(input_features)))

    print(f"Data baru berhasil diproses. Bentuk input untuk prediksi: {input_for_prediction.shape}")
    return input_for_prediction

# --- Fungsi Bantu untuk Membuat Prediksi dan Inverse Scaling ---
def make_prediction_and_inverse_scale(model, processed_input, scaler_target):
    """Membuat prediksi menggunakan model dan mengembalikan hasilnya ke skala asli."""
    print("Membuat prediksi...")
    # Prediksi menggunakan model
    prediction_scaled = model.predict(processed_input)
    print(f"Hasil prediksi (scaled): {prediction_scaled}")

    # Inverse scaling hasil prediksi menggunakan scaler target yang sudah dilatih
    # scaler_target.inverse_transform() butuh input 2D array (samples, features)
    prediction_original_scale = scaler_target.inverse_transform(prediction_scaled)

    print("Inverse scaling berhasil.")
    print(f"Hasil prediksi (skala harga asli - HLC): {prediction_original_scale[0]}") # [0] karena hanya 1 sampel

    return prediction_original_scale[0] # Mengembalikan array 1D [High, Low, Close]

# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- 1. Muat Model dan Scaler ---
    model, scaler_input, scaler_target = load_model_and_scalers(MODEL_PATH, SCALER_INPUT_PATH, SCALER_TARGET_PATH)

    if model is None or scaler_input is None or scaler_target is None:
        print("Tidak bisa melanjutkan karena model atau scaler gagal dimuat.")
    else:
        # --- 2. Dapatkan Data Baru ---
        # BAGIAN INI PERLU ANDA SESUAIKAN
        # Di sini adalah placeholder. Anda perlu menggantinya dengan kode
        # untuk mendapatkan data OHLCV terbaru dari sumber data Anda (misal: API, database).
        # Pastikan data baru yang Anda dapatkan memiliki panjang minimal WINDOW_SIZE
        # dan memiliki kolom 'Open', 'High', 'Low', 'Close', 'Volume'.
        print("\nMendapatkan data baru (Placeholder)...")
        # Contoh placeholder: membuat DataFrame dummy (ganti dengan data real Anda)
        # Data dummy ini hanya contoh, Anda perlu menggantinya dengan data OHLCV NYATA
        # yang cukup panjang (minimal 256 titik data terbaru)
        dummy_data = {
            'Timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=WINDOW_SIZE, freq='15min')),
            'Open': np.random.rand(WINDOW_SIZE) * 100 + 1500,
            'High': np.random.rand(WINDOW_SIZE) * 100 + 1550,
            'Low': np.random.rand(WINDOW_SIZE) * 100 + 1450,
            'Close': np.random.rand(WINDOW_SIZE) * 100 + 1500,
            'Volume': np.random.rand(WINDOW_SIZE) * 1000,
            'Sentiment': ['neutral'] * WINDOW_SIZE # Sentimen tidak digunakan untuk input model ini
        }
        new_data_df = pd.DataFrame(dummy_data)
        print(f"Data baru dummy dibuat dengan {len(new_data_df)} titik data.")
        # Pastikan data baru terurut berdasarkan waktu (penting!)
        new_data_df.sort_values(by='Timestamp', inplace=True)


        # --- 3. Proses Data Baru ---
        processed_input = preprocess_new_data(new_data_df, scaler_input, WINDOW_SIZE, INPUT_FEATURES)

        if processed_input is not None:
            # --- 4. Buat Prediksi dan Inverse Scaling ---
            predicted_hlc_original_scale = make_prediction_and_inverse_scale(model, processed_input, scaler_target)

            print("\n--- Hasil Prediksi Akhir (Skala Harga Asli) ---")
            print(f"Prediksi HLC (High, Low, Close) untuk langkah waktu berikutnya: {predicted_hlc_original_scale}")

            # --- Anda bisa menambahkan logika di sini untuk menggunakan hasil prediksi ---
            # Contoh: Menyimpan hasil ke database, mengirim notifikasi, dll.
            # print("\nMenyimpan hasil prediksi atau menggunakannya...")


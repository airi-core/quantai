import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime # Untuk potensi sorting data jika belum urut

# --- Konfigurasi ---
# Jika file diupload langsung ke sesi Colab:
FILE_PATH = 'XAU_15m_data_processed.json'
# Jika file ada di Google Drive dan Drive sudah di-mount:
# from google.colab import drive
# drive.mount('/content/drive')
# FILE_PATH = '/content/drive/My Drive/path/to/your/XAU_15m_data_processed.json' # Sesuaikan path-nya

WINDOW_SIZE = 256
TRAIN_SPLIT_RATIO = 0.8 # 80% data untuk training
VAL_SPLIT_RATIO = 0.1   # 10% data untuk validation (sisanya 10% untuk test)
# Fitur yang akan digunakan sebagai input (OHLCV) - sesuai permintaan Anda
INPUT_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
# Fitur yang akan diprediksi sebagai output (HLC langkah berikutnya) - sesuai permintaan Anda
TARGET_FEATURES = ['High', 'Low', 'Close']

# --- Langkah 1: Load Data ---
print(f"Meload data dari {FILE_PATH}...")
try:
    with open(FILE_PATH, 'r') as f:
        raw_data = json.load(f)
    print("File berhasil dibaca.")
except FileNotFoundError:
    print(f"ERROR: File tidak ditemukan di {FILE_PATH}. Pastikan file sudah diupload atau Drive sudah di-mount dan path-nya benar.")
    # Keluar dari script jika file tidak ditemukan
    exit()
except Exception as e:
    print(f"ERROR saat membaca file: {e}")
    exit()


# --- Langkah 2: Pembersihan & Parsing ke DataFrame ---
# Header dari data Anda: ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']
# Baris pertama adalah header yang salah, kita ambil data mulai dari indeks 1
if isinstance(raw_data, list) and len(raw_data) > 1 and isinstance(raw_data[0], list):
    data_rows = raw_data[1:]
    # Buat DataFrame, asumsikan urutan kolom sudah sesuai cuplikan data
    # 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment'
    df = pd.DataFrame(data_rows, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment'])

    # Konversi kolom numerik dan handle null
    # Menggunakan errors='coerce' akan mengubah nilai non-numerik menjadi NaN
    for col in INPUT_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Konversi Timestamp (opsional, tapi baik untuk memastikan urutan)
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    # Hapus baris dengan Timestamp yang tidak valid (jika ada setelah coerce)
    # df.dropna(subset=['Timestamp'], inplace=True)
    # Pastikan data terurut berdasarkan waktu (jika belum pasti)
    # df.sort_values(by='Timestamp', inplace=True)
    # df.reset_index(drop=True, inplace=True)

    # Hapus baris dengan nilai NaN di fitur input yang relevan
    df.dropna(subset=INPUT_FEATURES, inplace=True)
    print(f"Data berhasil diload dan dibersihkan. Jumlah baris data: {len(df)}")
    print("Cuplikan data setelah dibersihkan:")
    print(df.head())

else:
    print("ERROR: Format data JSON tidak sesuai perkiraan (bukan list of lists).")
    exit()


# --- Langkah 3: Pemilihan Fitur Input (OHLCV) ---
data_input = df[INPUT_FEATURES].values # Ambil nilai OHLCV dalam bentuk numpy array
print(f"Bentuk data input mentah (OHLCV): {data_input.shape}")

# --- Langkah 4: Normalisasi/Scaling Data ---
# Sangat PENTING untuk LSTM! Latih scaler HANYA pada data input (fitur OHLCV)
# Kita akan scale target (HLC) secara terpisah nanti jika perlu, atau scale semua bersama lalu inverse terpisah.
# Mari kita scale input dan target secara terpisah untuk kejelasan.

# Scaler untuk fitur INPUT (OHLCV)
scaler_input = MinMaxScaler()
data_input_scaled = scaler_input.fit_transform(data_input)

# Siapkan data target mentah (HLC) - geser data ke atas untuk mendapatkan target 'berikutnya'
# Target untuk baris i adalah HLC dari baris i+1
# Baris terakhir tidak punya target, jadi kita abaikan
data_target_raw = df[TARGET_FEATURES].values
# data_target_shifted = np.roll(data_target_raw, shift=-1, axis=0) # Geser ke atas 1 baris
# # Hapus baris terakhir karena targetnya sekarang adalah baris pertama (yang sudah digeser)
# data_target_shifted = data_target_shifted[:-1]
# # Hapus baris terakhir dari data input scaled juga agar jumlahnya sama
# data_input_scaled_for_seq = data_input_scaled[:-1]

# Koreksi logika pembuatan target: target untuk window yang berakhir di index i-1 adalah nilai HLC di index i.
# Jadi, data input akan dari index 0 hingga N-1, dan target akan dari index 1 hingga N.
# Jumlah sampel yang bisa dibuat adalah len(data) - WINDOW_SIZE.
# Input window berakhir di index i + WINDOW_SIZE - 1
# Target berada di index i + WINDOW_SIZE

# Scaler untuk fitur TARGET (HLC) - latih pada data_target_raw *sebelum* digeser
scaler_target = MinMaxScaler()
scaler_target.fit(data_target_raw) # Latih scaler pada SEMUA data target asli

# --- Langkah 5: Pembuatan Jendela Sekuensial (Create Sequences) ---
X, y = [], []
# Loop dari indeks 0 hingga indeks terakhir yang memungkinkan untuk membuat window + target
# Indeks terakhir untuk awal window adalah len(data_input_scaled) - WINDOW_SIZE - 1
num_samples = len(data_input_scaled) - WINDOW_SIZE

print(f"Membuat sequence dengan window size {WINDOW_SIZE}. Jumlah sampel yang akan dibuat: {num_samples}")

if num_samples < 0:
    print("ERROR: Jumlah data tidak cukup untuk window size ini!")
else:
    for i in range(num_samples):
        # Input adalah window dari data_input_scaled
        # Window dimulai dari index i dan berakhir di index i + WINDOW_SIZE - 1
        input_window = data_input_scaled[i : i + WINDOW_SIZE]

        # Target adalah data_target_raw di index i + WINDOW_SIZE
        # Ini adalah HLC dari langkah waktu T+1 jika window berakhir di T
        target_value_raw = data_target_raw[i + WINDOW_SIZE]

        # Scale target value
        target_value_scaled = scaler_target.transform([target_value_raw])[0] # Transform butuh 2D array, [0] untuk ambil hasilnya

        X.append(input_window)
        y.append(target_value_scaled)

    X = np.array(X)
    y = np.array(y)

    print(f"Bentuk array input X (sekuens): {X.shape}")
    print(f"Bentuk array target y: {y.shape}")

    # --- Langkah 6: Pembagian Data (Train/Validation/Test) ---
    # Pembagian Kronologis
    total_samples = len(X)
    train_size = int(total_samples * TRAIN_SPLIT_RATIO)
    val_size = int(total_samples * VAL_SPLIT_RATIO)
    test_size = total_samples - train_size - val_size

    X_train, y_train = X[ :train_size], y[ :train_size]
    X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]
    X_test, y_test = X[train_size + val_size : ], y[train_size + val_size : ]

    print("\n--- Bentuk Data Setelah Pembagian ---")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # --- Langkah 7: Reshape Data Input (sudah dilakukan oleh np.array) ---
    # Array numpy X sudah memiliki shape (jumlah_sampel, window, fitur)
    # Pastikan bentuknya (jumlah_sampel, 256, 5)

    print("\nPersiapan data selesai. Data siap untuk pembangunan dan pelatihan model.")

    # Opsional: Simpan scaler jika perlu untuk inverse transform prediksi nanti
    # import joblib
    # joblib.dump(scaler_input, 'scaler_input.pkl')
    # joblib.dump(scaler_target, 'scaler_target.pkl')
    # print("Scalers disimpan.")


# -*- coding: utf-8 -*-
"""
Kode Pipeline QuantAI End-to-End dalam Satu File.

Kode ini mengimplementasikan seluruh alur kerja ML untuk model QuantAI,
dari pemuatan data mentah hingga penyimpanan model terlatih dan hasilnya.
Dirancang untuk dijalankan di lingkungan standar Python,
termasuk runner GitHub Actions.

Fitur Termasuk:
- Membaca konfigurasi dari file YAML (path diberikan sebagai argumen command-line).
- Setup TensorFlow dan Hardware (GPU/Mixed Precision).
- Pemuatan data (CSV/JSON) dan preprocessing awal (termasuk Feature Engineering aritmatika).
- Pembuatan fitur teks deskriptif.
- Scaling data numerik.
- Pembuatan pipeline tf.data yang efisien (windowing, batching, caching, prefetching),
  termasuk penanganan data teks.
- Definisi atau pemuatan model QuantAI (Functional API dengan GRU, LayerNorm, Dense,
  potensi input teks).
- Kompilasi model.
- Pelatihan model (mode initial_train atau incremental_learn) dengan callbacks.
- Evaluasi model akhir.
- Penyimpanan model terlatih (SavedModel), scaler, hasil evaluasi, dan prediksi.
- Penggunaan API aritmatika, baca/tulis, dan fondasi belajar mandiri/hybrid.

Untuk menjalankan di GitHub Actions:
1. Pastikan kode ini dan quantai_config.yaml ada di repositori.
2. Buat file requirements.txt dengan dependensi yang dibutuhkan.
3. Buat file workflow GitHub Actions (.github/workflows/train_pipeline.yml).
4. Workflow akan menjalankan script ini dengan path ke file config sebagai argumen.
"""

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
import json # Untuk menyimpan hasil evaluasi/prediksi dalam JSON
import argparse # Untuk membaca argumen command-line

# --- Konfigurasi Global ---
# Path ke file konfigurasi akan diberikan melalui argumen command-line
# CONFIG_PATH = 'quantai_config.yaml' # Tidak lagi hardcoded di sini

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Fungsi Pembantu ---

def generate_analysis_text(row):
    """
    Menghasilkan string teks deskriptif berdasarkan nilai numerik baris data.
    Menggunakan operasi aritmatika dan perbandingan.
    """
    text = []
    # Contoh logika sederhana menggunakan nilai Pivot dan harga
    pivot = row.get('Pivot', None)
    close = row.get('Close', None)
    r1 = row.get('R1', None)
    s1 = row.get('S1', None)
    high = row.get('High', None)
    low = row.get('Low', None)

    # Pastikan nilai ada dan bukan NaN sebelum perbandingan/aritmatika
    if pivot is not None and close is not None and not pd.isna(pivot) and not pd.isna(close):
        if close > pivot:
            text.append(f"Close {close:.2f} di atas Pivot {pivot:.2f}, bullish.")
        elif close < pivot:
            text.append(f"Close {close:.2f} di bawah Pivot {pivot:.2f}, bearish.")
        else:
            text.append(f"Close {close:.2f} di sekitar Pivot {pivot:.2f}, netral.")

    if r1 is not None and high is not None and not pd.isna(r1) and not pd.isna(high):
         # Menggunakan Aritmatika untuk perbandingan
        if high > r1:
            text.append(f"High {high:.2f} menembus R1 {r1:.2f}, potensi naik.")

    if s1 is not None and low is not None and not pd.isna(s1) and not pd.isna(low):
         # Menggunakan Aritmatika untuk perbandingan
        if low < s1:
            text.append(f"Low {low:.2f} menembus S1 {s1:.2f}, potensi turun.")

    # Tambahkan logika lain sesuai kebutuhan (misal: R2-R5, S2-S5, volatilitas)

    return " ".join(text) if text else "Analisis tidak tersedia." # Gabungkan semua frasa


# Menggunakan @tf.function untuk mengompilasi fungsi pemrosesan teks di pipeline tf.data
@tf.function
def process_text_for_model(text_tensor, max_sequence_length, vocabulary_table):
    """
    Memproses tensor string teks menjadi representasi numerik untuk model.
    Contoh: tokenisasi, lookup ID, padding.
    """
    # Menggunakan API TensorFlow Strings
    tokens = tf.strings.split(text_tensor) # Returns RaggedTensor

    # Menggunakan API TensorFlow Lookup
    # Converting string tokens to integer IDs
    token_ids = vocabulary_table.lookup(tokens) # Returns RaggedTensor

    # Padding or truncating the sequence of token IDs
    # Gunakan tf.ragged.to_tensor() untuk mengatasi AttributeError pada SymbolicTensor
    padded_token_ids = tf.ragged.to_tensor(token_ids, default_value=0, shape=(max_sequence_length,))

    return padded_token_ids

# Fungsi pemrosesan elemen dataset (untuk dipanggil di .map())
@tf.function # Keep this decorator for the main element processing function
def process_dataset_element(inputs, target_elem):
    """Memproses elemen dataset (termasuk teks) sebelum windowing."""
    processed_text = None # Default if text is not used
    # Cek apakah input adalah tuple (numeric, text) atau hanya numeric
    if isinstance(inputs, tuple) and len(inputs) == 2:
        numeric_elem, text_elem = inputs
        # Process text using the corrected function
        processed_text = process_text_for_model(text_elem, config['data']['max_text_sequence_length'], vocabulary_table)
        processed_inputs = (numeric_elem, processed_text)
    else:
        numeric_elem = inputs
        processed_inputs = numeric_elem

    return processed_inputs, target_elem # Mengembalikan tuple (inputs yang diproses, target)


def create_tf_dataset(dataset_raw, window_size, window_shift, drop_remainder, use_text_input_actual, max_text_sequence_length, vocabulary_table, shuffle=False, batch_size=None):
    """
    Membuat pipeline tf.data dengan pemrosesan elemen, windowing, batching, dll.
    """
    # Terapkan pemrosesan elemen (termasuk teks) sebelum windowing
    # Gunakan use_text_input_actual untuk menentukan apakah input adalah tuple atau tidak
    dataset_processed_elements = dataset_raw.map(
        lambda inputs, target_elem: process_dataset_element(inputs, target_elem),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Kemudian terapkan windowing
    dataset_windowed = dataset_processed_elements.window(size=window_size, shift=window_shift, drop_remainder=drop_remainder)

    # Kemudian flat_map untuk menggabungkan window dan batch
    dataset_batched_windows = dataset_windowed.flat_map(lambda window: window.batch(window_size))

    # Terapkan shuffle jika diminta (hanya untuk pelatihan)
    if shuffle:
        dataset_batched_windows = dataset_batched_windows.shuffle(config['data']['shuffle_buffer_size'])

    # Terapkan batching akhir
    if batch_size:
        dataset_batched_windows = dataset_batched_windows.batch(batch_size)

    # Cache dan prefetch
    dataset_batched_windows = dataset_batched_windows.cache() # Cache setelah windowing dan batching window
    dataset_batched_windows = dataset_batched_windows.prefetch(tf.data.AUTOTUNE)

    return dataset_batched_windows


def build_quantai_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size=None):
    """
    Membangun arsitektur model QuantAI menggunakan Functional API.
    Mendukung input numerik dan teks.
    """
    # Menggunakan API Keras Layers: Input
    numeric_input = tf.keras.layers.Input(shape=numeric_input_shape, name='numeric_input')

    # Jalur Pemrosesan Numerik
    x_numeric = numeric_input
    # Menggunakan API Keras Layers: GRU
    x_numeric = tf.keras.layers.GRU(config['model']['architecture']['gru_units'], return_sequences=True)(x_numeric)
    # Menggunakan API Keras Layers: LayerNormalization
    x_numeric = tf.keras.layers.LayerNormalization()(x_numeric)
    # Menggunakan API Keras Layers: Dropout
    x_numeric = tf.keras.layers.Dropout(config['model']['architecture']['dropout_rate'])(x_numeric)

    # Opsi: Tambahkan Conv1D untuk pola lokal (TCN-inspired element)
    if config['model']['architecture'].get('use_conv1d', False):
         # Menggunakan API Keras Layers: Conv1D
        x_numeric = tf.keras.layers.Conv1D(
            filters=config['model']['architecture']['conv1d_filters'],
            kernel_size=config['model']['architecture']['conv1d_kernel_size'],
            activation='relu', # Menggunakan API Aritmatika di dalam layer
            padding='causal', # Penting untuk deret waktu
            dilation_rate=config['model']['architecture'].get('conv1d_dilation_rate', 1) # Untuk TCN
        )(x_numeric)
         # Menggunakan API Keras Layers: LayerNormalization
        x_numeric = tf.keras.layers.LayerNormalization()(x_numeric)
         # Menggunakan API Keras Layers: Dropout
        x_numeric = tf.keras.layers.Dropout(config['model']['architecture']['dropout_rate'])(x_numeric)

    # Merangkum sekuens numerik
     # Menggunakan API Keras Layers: GlobalAveragePooling1D
    x_numeric = tf.keras.layers.GlobalAveragePooling1D()(x_numeric)


    # Jalur Pemrosesan Teks (jika digunakan)
    if config['use_text_input'] and vocabulary_size is not None:
        # Menggunakan API Keras Layers: Input
        text_input = tf.keras.layers.Input(shape=text_input_shape, name='text_input', dtype=tf.int64) # Input ID integer

        # Menggunakan API Keras Layers: Embedding
        # Menggunakan API Aritmatika di dalam layer Embedding
        x_text = tf.keras.layers.Embedding(input_dim=vocabulary_size,
                                           output_dim=config['model']['architecture']['embedding_dim'],
                                           mask_zero=True # Penting jika ada padding nol
                                          )(text_input)

        # Opsi: Tambahkan GRU atau Conv1D pada embedding teks
         # Menggunakan API Keras Layers: GRU
        x_text = tf.keras.layers.GRU(config['model']['architecture']['gru_units'] // 2)(x_text) # Setengah unit GRU
         # Menggunakan API Keras Layers: LayerNormalization
        x_text = tf.keras.layers.LayerNormalization()(x_text)
         # Menggunakan API Keras Layers: Dropout
        x_text = tf.keras.layers.Dropout(config['model']['architecture']['dropout_rate'])(x_text)

        # Menggabungkan jalur numerik dan teks
         # Menggunakan API Keras Layers: Concatenate
        combined = tf.keras.layers.Concatenate()([x_numeric, x_text])
    else:
        combined = x_numeric # Hanya jalur numerik jika teks tidak digunakan

    # Layer Akhir
     # Menggunakan API Keras Layers: Dense
    output_layer = tf.keras.layers.Dense(num_target_features, name='output')(combined) # 3 unit untuk HLC

    # Menggunakan API Keras Model: Functional API
    if config['use_text_input'] and vocabulary_size is not None:
        model = tf.keras.Model(inputs=[numeric_input, text_input], outputs=output_layer)
    else:
        model = tf.keras.Model(inputs=numeric_input, outputs=output_layer)

    return model

# --- Main Pipeline Function ---

def run_pipeline(config):
    """
    Menjalankan seluruh pipeline ML berdasarkan konfigurasi.
    Ini adalah inti dari script satu file.
    """
    logger.info("Memulai pipeline QuantAI...")

    # --- Langkah 0.5: Setup Hardware (Dilakukan di runner GitHub Actions) ---
    # Konfigurasi TensorFlow dan Hardware:
    try:
        logger.info("Mengkonfigurasi TensorFlow dan Hardware...")
        # Deteksi GPU:
        physical_devices_gpu = tf.config.list_physical_devices('GPU')
        if physical_devices_gpu:
            logger.info(f"Ditemukan GPU: {len(physical_devices_gpu)}")
            # Setel Perangkat Terlihat (Gunakan semua GPU yang terdeteksi):
            tf.config.set_visible_devices(physical_devices_gpu, 'GPU')
            # Atur Memori Growth untuk setiap GPU:
            for gpu in physical_devices_gpu:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth diaktifkan untuk GPU.")
            # Aktifkan Mixed Precision jika menggunakan GPU:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision (mixed_float16) diaktifkan.")
        else:
            logger.warning("Tidak ada GPU yang terdeteksi. Menggunakan CPU.")
            # Setel Perangkat Terlihat (Hanya gunakan CPU jika tidak ada GPU):
            physical_devices_cpu = tf.config.list_physical_devices('CPU')
            tf.config.set_visible_devices(physical_devices_cpu, 'CPU')
            # Mixed precision tidak relevan untuk CPU

        # Setel seed untuk reproduksibilitas:
        logger.info(f"Menyetel seed: {config['seed']}")
        np.random.seed(config['seed'])
        tf.random.set_seed(config['seed'])
        logger.info("Setup hardware dan seed selesai.")

    except Exception as e:
        logger.warning(f"Gagal mengkonfigurasi hardware TensorFlow: {e}")
        # Pipeline akan tetap berjalan di CPU jika konfigurasi GPU gagal


    # --- Langkah 1: Pemuatan Data & Preprocessing Awal ---
    logger.info("Langkah 1: Memuat data dan preprocessing awal...")
    df = None # Inisialisasi df
    try:
        # Menggunakan Pandas untuk membaca data (Membaca)
        data_path = config['data']['raw_path']
        if not os.path.exists(data_path):
             logger.error(f"File data tidak ditemukan di {data_path}. Pastikan path benar dan file ada di repositori/runner.")
             return # Hentikan pipeline jika file data tidak ada

        if data_path.endswith('.csv'):
            # Gunakan argumen sep=';' untuk membaca file CSV dengan pemisah titik koma
            # Tambahkan argumen decimal='.' karena header menunjukkan titik sebagai pemisah desimal
            # Jika data Anda menggunakan koma sebagai pemisah desimal, ganti decimal='.' menjadi decimal=','
            df = pd.read_csv(data_path, sep=';', decimal='.')
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Format data tidak didukung. Gunakan .csv atau .json.")

        logger.info(f"Data mentah dimuat dari {data_path}. Jumlah baris: {len(df)}")

        # Diagnostik: Log nama kolom dan tipe data setelah membaca CSV
        logger.info(f"Kolom setelah membaca CSV: {df.columns.tolist()}")
        logger.info(f"Tipe data setelah membaca CSV: {df.dtypes}")


        # Pembersihan data awal (menghapus baris dengan NaN)
        # Jangan dropna dulu, lakukan pembersihan string pada kolom harga dulu
        # df.dropna(inplace=True)
        # if len(df) < initial_rows:
        #     logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN.")

        # Pastikan kolom harga adalah numerik
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] # Tambahkan Volume
        for col in price_cols:
            if col in df.columns:
                # PERBAIKAN: Bersihkan string dari karakter non-numerik sebelum konversi
                # Hapus spasi di awal/akhir
                df[col] = df[col].astype(str).str.strip()
                # Hapus koma di akhir (jika ada)
                df[col] = df[col].astype(str).str.rstrip(',')
                # Argumen decimal='.' di pd.read_csv sudah menangani pemisah desimal titik

                # Menggunakan Pandas to_numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                 # Diagnostik: Log jika kolom harga tidak ditemukan setelah membaca CSV
                 logger.warning(f"Kolom harga '{col}' tidak ditemukan di DataFrame setelah membaca CSV.")


        # Hapus baris jika kolom harga jadi NaN setelah konversi (karena pembersihan tidak sempurna atau memang ada data non-numerik)
        initial_rows = len(df)
        df.dropna(subset=price_cols, inplace=True)
        if len(df) < initial_rows:
             logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di kolom harga setelah konversi numerik.")

        # Jika setelah pembersihan DataFrame kosong, hentikan
        if df.empty:
             logger.error("DataFrame kosong setelah pembersihan data awal. Tidak dapat melanjutkan.")
             return # Hentikan pipeline jika data kosong

        # Diagnostik: Log nama kolom dan tipe data setelah pembersihan dan konversi numerik
        logger.info(f"Kolom setelah pembersihan numerik: {df.columns.tolist()}")
        logger.info(f"Tipe data setelah pembersihan numerik: {df.dtypes}")


        # Feature Engineering (Menggunakan Aritmatika)
        # Contoh: Menghitung Pivot Points
        logger.info("Menghitung Pivot Points...")
        # Menggunakan Aritmatika Pandas Series
        # Pastikan kolom yang dibutuhkan ada sebelum diakses
        required_pivot_cols = ['High', 'Low', 'Close']
        if all(col in df.columns for col in required_pivot_cols):
             df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
             df['R1'] = 2 * df['Pivot'] - df['Low']
             df['S1'] = 2 * df['Pivot'] - df['High']
             df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
             df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
             df['R3'] = df['Pivot'] + 2 * (df['High'] - df['Low'])
             df['S3'] = df['Pivot'] - 2 * (df['High'] - df['Low'])
             # Tambahkan R4/R5, S4/S5 jika perlu
             logger.info("Pivot Points dan Support/Resistance dihitung.")
        else:
             logger.warning(f"Kolom yang dibutuhkan untuk Pivot Points ({required_pivot_cols}) tidak lengkap. Tidak dapat menghitung indikator.")
             # Jika indikator tidak bisa dihitung, pastikan feature_cols_numeric tidak menyertakannya jika tidak ada
             # Ini bisa menjadi sumber error jika feature_cols_numeric di config menyertakan indikator yang tidak ada

        # Menyiapkan Target (HLC Selanjutnya)
        logger.info(f"Menyiapkan target HLC selanjutnya dengan shift {config['parameter_windowing']['window_size']}...")
        # Menggunakan Pandas shift
        required_target_cols_base = ['High', 'Low', 'Close']
        target_cols = ['High_Next', 'Low_Next', 'Close_Next']
        if all(col in df.columns for col in required_target_cols_base):
             df['High_Next'] = df['High'].shift(-config['parameter_windowing']['window_size'])
             df['Low_Next'] = df['Low'].shift(-config['parameter_windowing']['window_size'])
             df['Close_Next'] = df['Close'].shift(-config['parameter_windowing']['window_size'])
             logger.info("Target HLC selanjutnya disiapkan.")
        else:
             logger.warning(f"Kolom yang dibutuhkan untuk target ({required_target_cols_base}) tidak lengkap. Tidak dapat menyiapkan target.")
             # Jika target tidak bisa disiapkan, script akan berhenti di dropna target

        # Menghapus baris terakhir yang memiliki NaN setelah shift
        initial_rows = len(df)
        # Pastikan target_cols ada di df.columns sebelum dropna
        target_cols_present = [col for col in target_cols if col in df.columns]
        if target_cols_present:
             df.dropna(subset=target_cols_present, inplace=True)
             if len(df) < initial_rows:
                  logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di target setelah shift.")
        else:
             logger.warning("Tidak ada kolom target yang tersedia untuk dropna.")


        # Jika setelah menyiapkan target DataFrame kosong, hentikan
        if df.empty:
             logger.error("DataFrame kosong setelah menyiapkan target. Tidak dapat melanjutkan.")
             return # Hentikan pipeline jika data kosong


        # Membuat Fitur Teks Deskriptif (Menulis Teks ke Kolom)
        if config['use_text_input']:
            logger.info("Membuat fitur teks deskriptif...")
            # Menggunakan Pandas apply dan fungsi kustom generate_analysis_text
            # Pastikan kolom yang dibutuhkan oleh generate_analysis_text ada
            required_analysis_cols = ['Pivot', 'Close', 'R1', 'S1', 'High', 'Low']
            # Cek apakah semua kolom yang dibutuhkan untuk analisis teks ada di DataFrame
            if all(col in df.columns for col in required_analysis_cols):
                 df['Analysis_Text'] = df.apply(generate_analysis_text, axis=1)
                 if not df.empty:
                      logger.info(f"Contoh teks analisis: {df['Analysis_Text'].iloc[0]}")
                 else:
                      # Ini seharusnya sudah ditangani oleh cek df.empty di atas
                      logger.warning("DataFrame kosong saat membuat teks analisis.")
            else:
                 # Jika kolom untuk analisis teks tidak lengkap, nonaktifkan input teks
                 logger.warning(f"Kolom yang dibutuhkan untuk teks analisis ({required_analysis_cols}) tidak lengkap. Nonaktifkan input teks.")
                 config['use_text_input'] = False # Nonaktifkan input teks jika kolom tidak ada


        # Identifikasi Fitur Input dan Target
        # Pastikan kolom indikator yang dihitung juga masuk fitur numerik
        # Hanya sertakan indikator yang benar-benar ada di DataFrame
        indicator_cols = [col for col in df.columns if col.startswith('R') or col.startswith('S') or col == 'Pivot'] # Identifikasi ulang indikator yang ada
        indicator_cols_present = [col for col in indicator_cols if col in df.columns]
        # Gabungkan kolom numerik dari config dengan indikator yang ada
        # Gunakan set untuk menghindari duplikat dan list comprehension untuk mempertahankan urutan (opsional)
        feature_cols_numeric_base = config['data']['feature_cols_numeric']
        feature_cols_numeric = list(dict.fromkeys(feature_cols_numeric_base + indicator_cols_present)) # Gabungkan dan hapus duplikat

        # Filter feature_cols_numeric untuk hanya menyertakan kolom yang ada di DataFrame
        feature_cols_numeric_present = [col for col in feature_cols_numeric if col in df.columns]

        # Cek apakah 'Analysis_Text' berhasil dibuat sebelum menambahkannya ke feature_cols_text
        feature_cols_text = ['Analysis_Text'] if config.get('use_text_input', False) and 'Analysis_Text' in df.columns else []
        target_cols_present = [col for col in target_cols if col in df.columns] # Gunakan target_cols yang sudah disiapkan

        # Konversi ke NumPy Arrays
        # Menggunakan NumPy untuk konversi
        # Pastikan hanya kolom yang dipilih yang diambil
        if not df.empty and feature_cols_numeric_present and target_cols_present:
            data_numeric = df[feature_cols_numeric_present].values.astype(np.float32)
            data_text = df[feature_cols_text].values.flatten().astype(str) if feature_cols_text else np.array([], dtype=str)
            data_target = df[target_cols_present].values.astype(np.float32)

            logger.info(f"Data numerik shape: {data_numeric.shape}")
            logger.info(f"Data teks shape: {data_text.shape}")
            logger.info(f"Data target shape: {data_target.shape}")
        else:
            logger.error("DataFrame kosong atau kolom fitur/target tidak tersedia setelah preprocessing. Tidak dapat melanjutkan.")
            # Diagnostik: Log kolom yang tersedia vs yang dicari
            logger.error(f"Kolom tersedia di DataFrame: {df.columns.tolist()}")
            logger.error(f"Kolom numerik yang dicari: {feature_cols_numeric_present}")
            logger.error(f"Kolom teks yang dicari: {feature_cols_text}")
            logger.error(f"Kolom target yang dicari: {target_cols_present}")

            return # Hentikan pipeline jika data kosong atau kolom tidak lengkap


        # Membagi Data (menggunakan slicing NumPy)
        total_samples = len(df)
        train_size = int(total_samples * config['data']['train_split'])
        val_size = int(total_samples * config['data']['val_split'])
        test_size = total_samples - train_size - val_size # Ukuran test adalah sisanya

        if train_size <= 0 or val_size <= 0 or test_size <= 0:
             logger.error("Ukuran set data (train/val/test) terlalu kecil (<= 0). Sesuaikan split atau gunakan data lebih banyak.")
             return # Hentikan pipeline jika ukuran data tidak memadai


        input_train_numeric = data_numeric[:train_size]
        input_val_numeric = data_numeric[train_size:train_size + val_size]
        input_test_numeric = data_numeric[train_size + val_size:]

        input_train_text = data_text[:train_size] if feature_cols_text else np.array([], dtype=str)
        input_val_text = data_text[train_size:train_size + val_size] if feature_cols_text else np.array([], dtype=str)
        input_test_text = data_text[train_size + val_size:] if feature_cols_text else np.array([], dtype=str)

        target_train = data_target[:train_size]
        target_val = data_target[train_size:train_size + val_size]
        target_test = data_target[train_size + val_size:]

        logger.info(f"Train set size: {len(input_train_numeric)}")
        logger.info(f"Validation set size: {len(input_val_numeric)}")
        logger.info(f"Test set size: {len(input_test_numeric)}")

    except Exception as e:
        logger.error(f"Error selama Langkah 1: {e}")
        return # Hentikan pipeline jika ada error fatal

    # --- Langkah 2: Scaling Data & Pembuatan Pipeline tf.data ---
    logger.info("Langkah 2: Scaling data dan membuat pipeline tf.data...")
    scaler_input = None # Inisialisasi scaler
    scaler_target = None
    vocabulary_table = None
    vocabulary_size = None
    try:
        # Scaling Data Numerik (Menggunakan Aritmatika di dalam scaler)
        scaler_input = MinMaxScaler()
        scaler_target = MinMaxScaler()

        # Latih scaler hanya pada data pelatihan
        scaled_input_train = scaler_input.fit_transform(input_train_numeric)
        scaled_target_train = scaler_target.fit_transform(target_train)

        # Terapkan scaler pada data validasi dan pengujian
        scaled_input_val = scaler_input.transform(input_val_numeric)
        scaled_input_test = scaler_input.transform(input_test_numeric)

        scaled_target_val = scaler_target.transform(target_val)
        # target_test tidak perlu diskalakan di sini, hanya input_test_numeric yang diskalakan

        # Membuat Kosakata untuk Teks (jika digunakan)
        if feature_cols_text: # Cek apakah fitur teks benar-benar digunakan
            logger.info("Membuat kosakata dari data pelatihan teks...")
            # Mengumpulkan semua token unik dari data pelatihan teks
            # Menggunakan API TensorFlow Strings: split
            all_train_tokens_ragged = tf.strings.split(input_train_text)
            all_train_tokens = all_train_tokens_ragged.values.numpy() # Ambil nilai tensor datar
            unique_tokens = np.unique(all_train_tokens[all_train_tokens != b'']) # Hilangkan token kosong

            # Menggunakan API TensorFlow Lookup: StaticVocabularyTable
            # Membuat tabel lookup dari kosakata unik
            keys = tf.constant(unique_tokens)
            values = tf.range(tf.size(keys), dtype=tf.int64)
            init = tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int64)
            # num_oov_buckets=1: 1 bucket untuk token di luar kosakata
            vocabulary_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=1)
            vocabulary_size = vocabulary_table.size().numpy() + 1 # Ukuran kosakata + OOV bucket
            logger.info(f"Ukuran kosakata teks: {vocabulary_size}")
            # Decode untuk logging, handle jika unique_tokens kosong
            if unique_tokens.size > 0:
                 logger.info(f"Contoh token unik: {[t.decode('utf-8') for t in unique_tokens[:10]]}")
            else:
                 logger.warning("Tidak ada token unik yang ditemukan dalam data teks pelatihan. Input teks mungkin tidak efektif.")


        # Membuat tf.data.Dataset dari data yang sudah diproses
        # Menggunakan API TensorFlow Data: from_tensor_slices
        if feature_cols_text: # Cek apakah fitur teks benar-benar digunakan
            dataset_train_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_train, input_train_text), scaled_target_train))
            dataset_val_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_val, input_val_text), scaled_target_val))
            # Target test tidak diskalakan, jadi gunakan target_test asli
            dataset_test_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_test, input_test_text), target_test))
        else:
             # Menggunakan API TensorFlow Data: from_tensor_slices
            dataset_train_raw = tf.data.Dataset.from_tensor_slices((scaled_input_train, scaled_target_train))
            dataset_val_raw = tf.data.Dataset.from_tensor_slices((scaled_input_val, scaled_target_val))
            # Target test tidak diskalakan, jadi gunakan target_test asli
            dataset_test_raw = tf.data.Dataset.from_tensor_slices((scaled_input_test, target_test))


        # Fungsi pemrosesan elemen dataset (untuk dipanggil di .map())
        @tf.function # Keep this decorator for the main element processing function
        def process_dataset_element(inputs, target_elem):
            """Memproses elemen dataset (termasuk teks) sebelum windowing."""
            processed_text = None # Default if text is not used
            # Cek apakah input adalah tuple (numeric, text) atau hanya numeric
            if isinstance(inputs, tuple) and len(inputs) == 2:
                numeric_elem, text_elem = inputs
                # Process text using the corrected function
                processed_text = process_text_for_model(text_elem, config['data']['max_text_sequence_length'], vocabulary_table)
                processed_inputs = (numeric_elem, processed_text)
            else:
                numeric_elem = inputs
                processed_inputs = numeric_elem

            return processed_inputs, target_elem # Mengembalikan tuple (inputs yang diproses, target)


        # Fungsi untuk membuat pipeline tf.data dengan windowing, batching, dll.
        def create_tf_dataset(dataset_raw, window_size, window_shift, drop_remainder, use_text_input_actual, max_text_sequence_length, vocabulary_table, shuffle=False, batch_size=None):
            """
            Membuat pipeline tf.data dengan pemrosesan elemen, windowing, batching, dll.
            """
            # Terapkan pemrosesan elemen (termasuk teks) sebelum windowing
            # Gunakan use_text_input_actual untuk menentukan apakah input adalah tuple atau tidak
            dataset_processed_elements = dataset_raw.map(
                lambda inputs, target_elem: process_dataset_element(inputs, target_elem),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # Kemudian terapkan windowing
            dataset_windowed = dataset_processed_elements.window(size=window_size, shift=window_shift, drop_remainder=drop_remainder)

            # Kemudian flat_map untuk menggabungkan window dan batch
            dataset_batched_windows = dataset_windowed.flat_map(lambda window: window.batch(window_size))

            # Terapkan shuffle jika diminta (hanya untuk pelatihan)
            if shuffle:
                dataset_batched_windows = dataset_batched_windows.shuffle(config['data']['shuffle_buffer_size'])

            # Terapkan batching akhir
            if batch_size:
                dataset_batched_windows = dataset_batched_windows.batch(batch_size)

            # Cache dan prefetch
            dataset_batched_windows = dataset_batched_windows.cache() # Cache setelah windowing dan batching window
            dataset_batched_windows = dataset_batched_windows.prefetch(tf.data.AUTOTUNE)

            return dataset_batched_windows


        # Membuat Pipeline tf.data untuk Train, Val, Test
        # Menggunakan API TensorFlow Data: window
        # Menggunakan API TensorFlow Data: flat_map
        # Menggunakan API TensorFlow Data: shuffle
        # Menggunakan API TensorFlow Data: batch
        # Menggunakan API TensorFlow Data: cache
        # Menggunakan API TensorFlow Data: prefetch
        # Menggunakan API TensorFlow Data: AUTOTUNE

        # Pipeline Pelatihan
        dataset_train = create_tf_dataset(
            dataset_train_raw,
            config['parameter_windowing']['window_size'],
            config['parameter_windowing']['window_shift'],
            True, # drop_remainder
            bool(feature_cols_text), # Gunakan bool(feature_cols_text) untuk use_text_input_actual
            config['data']['max_text_sequence_length'],
            vocabulary_table,
            shuffle=True,
            batch_size=config['training']['batch_size']
        )

        # Pipeline Validasi (tanpa shuffle)
        dataset_val = create_tf_dataset(
            dataset_val_raw,
            config['parameter_windowing']['window_size'],
            config['parameter_windowing']['window_shift'],
            True, # drop_remainder
            bool(feature_cols_text), # Gunakan bool(feature_cols_text) untuk use_text_input_actual
            config['data']['max_text_sequence_length'],
            vocabulary_table,
            shuffle=False,
            batch_size=config['training']['batch_size']
        )

        # Pipeline Pengujian (tanpa shuffle, tanpa cache jika data test besar)
        # Cache data test hanya jika ukurannya relatif kecil
        dataset_test = create_tf_dataset(
            dataset_test_raw,
            config['parameter_windowing']['window_size'],
            config['parameter_windowing']['window_shift'],
            True, # drop_remainder
            bool(feature_cols_text), # Gunakan bool(feature_cols_text) untuk use_text_input_actual
            config['data']['max_text_sequence_length'],
            vocabulary_table,
            shuffle=False,
            batch_size=config['training']['batch_size']
        )


        logger.info("Pipeline tf.data selesai dibuat.")

    except Exception as e:
        logger.error(f"Error selama Langkah 2: {e}")
        return # Hentikan pipeline jika ada error fatal


    # --- Langkah 3: Definisi & Kompilasi Model ---
    logger.info("Langkah 3: Mendefinisikan atau memuat model...")
    model = None
    try:
        if config['mode'] == 'initial_train':
            logger.info("Mode: initial_train. Mendefinisikan model baru.")
            # Menggunakan API Keras Model dan Layers
            # Mendefinisikan model QuantAI baru
            numeric_input_shape = (config['parameter_windowing']['window_size'], len(feature_cols_numeric_present)) # Gunakan kolom yang benar-benar ada
            text_input_shape = (config['parameter_windowing']['window_size'], config['data']['max_text_sequence_length']) if feature_cols_text else None # Gunakan bool(feature_cols_text)
            num_target_features = len(target_cols_present) # Gunakan kolom target yang benar-benar ada

            model = build_quantai_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size)

            # Kompilasi Model
            logger.info("Mengkompilasi model.")
             # Menggunakan API Keras Optimizers
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
             # Menggunakan API Keras Losses
            loss_fn = tf.keras.losses.MeanAbsoluteError() # Menggunakan Aritmatika
             # Menggunakan API Keras Metrics
            metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()] # Menggunakan Aritmatika

            # Menggunakan API Keras Model: compile
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

            logger.info("Model baru berhasil didefinisikan dan dikompilasi.")
            model.summary(print_fn=logger.info) # Cetak ringkasan model ke log

        elif config['mode'] in ['incremental_learn', 'predict_only']:
            load_path = config['model']['load_path']
            logger.info(f"Mode: {config['mode']}. Memuat model dari {load_path}")
            # Menggunakan API Keras Models: load_model atau tf.saved_model.load
            # Memuat model QuantAI yang sudah ada
            if not os.path.exists(load_path):
                 logger.error(f"File atau direktori model tidak ditemukan di {load_path}. Pastikan path benar dan model telah disimpan sebelumnya.")
                 return # Hentikan pipeline jika model tidak ditemukan untuk dimuat

            try:
                 # Menggunakan API Keras Models: load_model
                # custom_objects diperlukan jika model menggunakan layer/fungsi kustom yang tidak standar Keras
                # Saat ini, kita hanya pakai layer standar, jadi tidak perlu custom_objects
                model = tf.keras.models.load_model(load_path)
                logger.info("Model berhasil dimuat menggunakan tf.keras.models.load_model.")
            except Exception as e_keras:
                logger.warning(f"Gagal memuat dengan tf.keras.models.load_model: {e_keras}. Mencoba tf.saved_model.load...")
                try:
                    # Menggunakan API TensorFlow SavedModel: load
                    model = tf.saved_model.load(load_path)
                    logger.info("Model berhasil dimuat menggunakan tf.saved_model.load.")
                    # Jika memuat dengan tf.saved_model.load, mungkin perlu kompilasi ulang secara manual
                    if config['mode'] == 'incremental_learn':
                         # Menggunakan API Keras Model: compile
                        logger.info("Mengkompilasi ulang model untuk incremental_learn.")
                         # Menggunakan API Keras Optimizers
                        optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
                         # Menggunakan API Keras Losses
                        loss_fn = tf.keras.losses.MeanAbsoluteError()
                         # Menggunakan API Keras Metrics
                        metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                        # Perlu memastikan objek yang dimuat memiliki metode compile jika dimuat dengan tf.saved_model.load
                        # Jika model asli dibuat dengan Functional API/Sequential, ini seharusnya ada
                        if hasattr(model, 'compile'):
                             model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
                             logger.info("Model berhasil dikompilasi ulang.")
                        else:
                             logger.warning("Model yang dimuat tidak memiliki metode compile. Tidak dapat melanjutkan pelatihan inkremental.")
                             if config['mode'] == 'incremental_learn':
                                 model = None # Set model ke None jika tidak bisa dikompilasi untuk incremental_learn


                except Exception as e_savedmodel:
                    logger.error(f"Gagal memuat model dari {load_path} dengan kedua metode: {e_savedmodel}")
                    model = None # Set model ke None jika gagal memuat

            if model is not None:
                logger.info("Model berhasil dimuat.")
                # model.summary(print_fn=logger.info) # Ringkasan mungkin tidak tersedia untuk model yang dimuat dengan tf.saved_model.load

        else:
            logger.error(f"Mode operasional tidak valid: {config['mode']}")
            return # Hentikan pipeline jika mode tidak valid

    except Exception as e:
        logger.error(f"Error selama Langkah 3: {e}")
        return # Hentikan pipeline jika ada error fatal

    if model is None:
         logger.error("Model tidak tersedia setelah Langkah 3. Menghentikan pipeline.")
         return

    # --- Langkah 4: Pelatihan Model (Belajar Mandiri/Hybrid/Otonom) ---
    if config['mode'] in ['initial_train', 'incremental_learn']:
        logger.info(f"Langkah 4: Memulai pelatihan model dalam mode {config['mode']}...")
        try:
            # Menggunakan API Keras Model: fit
            # Menggunakan API Keras Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

            # Pastikan direktori save_path dan tensorboard_log_dir sudah dibuat di Langkah 6
            # atau buat di sini jika tidak ingin menunggu Langkah 6
            output_dir = config['output']['base_dir']
            model_save_path_full = os.path.join(output_dir, config['output']['model_subdir'])
            tensorboard_log_dir_full = os.path.join(output_dir, config['output']['tensorboard_log_dir'])
            tf.io.gfile.makedirs(model_save_path_full)
            tf.io.gfile.makedirs(scaler_save_dir_full)
            tf.io.gfile.makedirs(os.path.dirname(eval_results_path_full)) # Pastikan dir untuk file eval
            tf.io.gfile.makedirs(os.path.dirname(predictions_path_full)) # Pastikan dir untuk file prediksi
            tf.io.gfile.makedirs(tensorboard_log_dir_full) # Pastikan dir untuk log TensorBoard


            callbacks = [
                 # Menggunakan API Keras Callbacks: EarlyStopping
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'], restore_best_weights=True),
                 # Menggunakan API Keras Callbacks: ModelCheckpoint
                tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path_full, monitor='val_loss', save_best_only=True, save_format='tf'), # Simpan dalam format SavedModel
                 # Menggunakan API Keras Callbacks: ReduceLROnPlateau (Menggunakan Aritmatika)
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['training']['lr_reduce_factor'], patience=config['training']['lr_reduce_patience'], min_lr=config['training']['min_lr']),
                 # Menggunakan API Keras Callbacks: TensorBoard
                tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir_full)
            ]

            epochs_to_run = config['training']['epochs'] if config['mode'] == 'initial_train' else config['training']['incremental_epochs']

            logger.info(f"Melatih selama {epochs_to_run} epoch.")
            history = model.fit(
                dataset_train,
                epochs=epochs_to_run,
                validation_data=dataset_val,
                callbacks=callbacks
            )
            logger.info("Pelatihan selesai.")

            # Muat kembali model terbaik setelah pelatihan selesai (ModelCheckpoint menyimpannya)
            # Ini penting jika EarlyStopping menghentikan pelatihan sebelum epoch terakhir
            logger.info(f"Memuat model terbaik dari {model_save_path_full}")
             # Menggunakan API Keras Models: load_model
            # custom_objects mungkin diperlukan di sini jika model menggunakan komponen kustom
            try:
                model = tf.keras.models.load_model(model_save_path_full)
                logger.info("Model terbaik berhasil dimuat.")
            except Exception as e:
                 logger.warning(f"Gagal memuat model terbaik setelah pelatihan dari {model_save_path_full}: {e}. Menggunakan model akhir dari fit().")
                 # Jika gagal memuat model terbaik, gunakan model yang ada setelah fit()


            # --- Fondasi Belajar Mandiri/Hybrid/Otonom (Loop Kustom Opsional) ---
            # Jika diperlukan logika update bobot yang lebih granular dari model.fit,
            # implementasikan loop kustom di sini menggunakan:
            # @tf.function
            # tf.GradientTape
            # optimizer.apply_gradients
            # model.trainable_variables
            # tf.cond, tf.while_loop (untuk logika otonom dalam grafik)
            # tf.py_function (untuk logika otonom di Python)
            # train_on_batch (untuk update per batch dalam loop Python)
            # Contoh (pseudocode):
            # @tf.function
            # def custom_train_step(inputs, targets):
            #     with tf.GradientTape() as tape:
            #         predictions = model(inputs, training=True)
            #         loss = model.compiled_loss(targets, predictions) # Menggunakan loss yang dikompilasi
            #         # Tambahkan loss kustom jika ada: loss += sum(model.losses)
            #     gradients = tape.gradient(loss, model.trainable_variables)
            #     model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Menggunakan optimizer yang dikompilasi
            #     # Update metrik kompilasi jika ada: model.compiled_metrics.update_state(targets, predictions)
            #     return loss
            #
            # if config['mode'] == 'incremental_learn_custom': # Mode kustom baru
            #     logger.info("Memulai pelatihan inkremental dengan loop kustom.")
            #     # Pastikan dataset_new_data disiapkan di Langkah 2
            #     if 'dataset_new_data' in locals():
            #         for epoch in range(config['training']['incremental_epochs_custom']):
            #             logger.info(f"Epoch inkremental kustom {epoch+1}/{config['training']['incremental_epochs_custom']}")
            #             # Reset metrik jika menggunakan metrik kompilasi
            #             # for metric in model.metrics: metric.reset_states()
            #             for batch_inputs, batch_targets in dataset_new_data:
            #                 batch_loss = custom_train_step(batch_inputs, batch_targets)
            #                 # Logika otonom: if batch_loss < threshold: ...
            #                 # Log loss atau metrik batch
            #             # Log metrik epoch jika menggunakan metrik kompilasi
            #             # eval_on_val_set() # Evaluasi pada set validasi secara berkala
            #         logger.info("Pelatihan inkremental kustom selesai.")
            #     else:
            #          logger.warning("Dataset data baru tidak tersedia untuk pelatihan inkremental kustom.")


        except Exception as e:
            logger.error(f"Error selama Langkah 4: {e}")
            # Lanjutkan ke langkah berikutnya meskipun ada error pelatihan,
            # mungkin kita masih ingin menyimpan model yang dimuat atau melakukan prediksi.


    # --- Langkah 5: Evaluation Model Akhir ---
    eval_results = None
    # Hanya lakukan evaluasi jika mode bukan 'predict_only'
    if config['mode'] in ['initial_train', 'incremental_learn']:
        logger.info("Langkah 5: Mengevaluasi model akhir...")
        try:
            # Menggunakan API Keras Model: evaluate
            # Menggunakan Aritmatika di dalam metrik
            # Pastikan model dikompilasi jika mode incremental_learn dan dimuat dengan tf.saved_model.load tanpa compile
            if not hasattr(model, 'evaluate') and hasattr(model, 'compile'):
                 logger.warning("Model tidak memiliki metode evaluate, mencoba mengkompilasi sebelum evaluasi.")
                 # Menggunakan API Keras Optimizers
                 optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
                 # Menggunakan API Keras Losses
                 loss_fn = tf.keras.losses.MeanAbsoluteError()
                 # Menggunakan API Keras Metrics
                 metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                 model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
                 logger.info("Model berhasil dikompilasi ulang untuk evaluasi.")


            if hasattr(model, 'evaluate') and 'dataset_test' in locals():
                eval_results = model.evaluate(dataset_test)
                logger.info(f"Hasil Evaluasi Akhir: {dict(zip(model.metrics_names, eval_results))}")
            elif 'dataset_test' not in locals():
                 logger.warning("Dataset test tidak tersedia. Tidak dapat melakukan evaluasi.")
            else:
                 logger.error("Model tidak memiliki metode evaluate setelah mencoba kompilasi. Tidak dapat melakukan evaluasi.")


        except Exception as e:
            logger.error(f"Error selama Langkah 5: {e}")
            # Lanjutkan ke langkah berikutnya meskipun ada error evaluasi

    # --- Langkah 6: Penyimpanan Aset & Hasil (Menulis) ---
    logger.info("Langkah 6: Menyimpan aset dan hasil...")
    try:
        # Pastikan direktori output ada (Menggunakan API TensorFlow IO GFile)
        output_dir = config['output']['base_dir']
        model_save_path_full = os.path.join(output_dir, config['output']['model_subdir'])
        scaler_save_dir_full = os.path.join(output_dir, config['output']['scaler_subdir'])
        eval_results_path_full = os.path.join(output_dir, config['output']['eval_results_file'])
        predictions_path_full = os.path.join(output_dir, config['output']['predictions_file'])
        tensorboard_log_dir_full = os.path.join(output_dir, config['output']['tensorboard_log_dir']) # Path ini digunakan oleh callback


        # Menggunakan API TensorFlow IO GFile: makedirs
        # Pastikan direktori dibuat sebelum menyimpan file di dalamnya
        tf.io.gfile.makedirs(model_save_path_full)
        tf.io.gfile.makedirs(scaler_save_dir_full)
        tf.io.gfile.makedirs(os.path.dirname(eval_results_path_full)) # Pastikan dir untuk file eval
        tf.io.gfile.makedirs(os.path.dirname(predictions_path_full)) # Pastikan dir untuk file prediksi
        tf.io.gfile.makedirs(tensorboard_log_dir_full) # Pastikan dir untuk log TensorBoard


        if config['mode'] in ['initial_train', 'incremental_learn']:
            # Menyimpan Model Terlatih (Hasil Pelatihan, Jalan Offline/Deploy Anywhere)
            logger.info(f"Menyimpan model terlatih ke {model_save_path_full}...")
             # Menggunakan API TensorFlow SavedModel: save
            # Jika model dimuat dengan tf.saved_model.load, mungkin perlu dikonversi ke Keras Model dulu
            # atau pastikan objek yang dimuat memang bisa disimpan sebagai SavedModel
            if isinstance(model, tf.keras.Model):
                 tf.saved_model.save(model, model_save_path_full)
                 logger.info("Model SavedModel berhasil disimpan.")
            else:
                 logger.warning("Objek model bukan tf.keras.Model. Tidak dapat menyimpan dalam format SavedModel standar.")


            # Menyimpan Scaler (Pendukung Offline/Deploy Anywhere)
            logger.info(f"Menyimpan scaler ke {scaler_save_dir_full}...")
             # Menggunakan Joblib: dump
            # Pastikan scaler_input dan scaler_target tersedia (dilatih di Langkah 2)
            if 'scaler_input' in locals() and 'scaler_target' in locals():
                 joblib.dump(scaler_input, os.path.join(scaler_save_dir_full, 'scaler_input.pkl'))
                 joblib.dump(scaler_target, os.path.join(scaler_save_dir_full, 'scaler_target.pkl'))
                 logger.info("Scaler berhasil disimpan.")
            else:
                 logger.warning("Scaler tidak tersedia. Tidak dapat menyimpan scaler.")


            # Menyimpan Hasil Evaluasi (Hasil Output)
            if eval_results is not None:
                logger.info(f"Menyimpan hasil evaluasi ke {eval_results_path_full}...")
                # Pastikan model.metrics_names tersedia jika eval_results bukan None
                if hasattr(model, 'metrics_names'):
                    eval_dict = dict(zip(model.metrics_names, eval_results))
                    # Menggunakan Python Standard Library: open dan json.dump (Menulis)
                    with open(eval_results_path_full, 'w') as f:
                        json.dump(eval_dict, f, indent=4)
                    logger.info("Hasil evaluasi berhasil disimpan.")
                else:
                     logger.warning("Nama metrik model tidak tersedia. Tidak dapat menyimpan hasil evaluasi dalam format dictionary.")


        # Menyimpan Prediksi (Hasil Output, Opsional untuk semua mode)
        # Hanya simpan prediksi jika mode bukan 'initial_train' atau 'incremental_learn' DAN save_predictions True
        # ATAU jika mode adalah 'predict_only' dan save_predictions True
        if config['output'].get('save_predictions', False) and model is not None:
             # Tentukan dataset mana yang akan diprediksi berdasarkan mode
             dataset_to_predict = None
             if config['mode'] == 'predict_only':
                 # Dalam mode predict_only, asumsikan dataset_test_raw adalah data yang akan diprediksi
                 # Perlu membuat pipeline predict_only dari dataset_test_raw
                 # Ini memerlukan sedikit penyesuaian di Langkah 2 untuk menyiapkan dataset_test_raw
                 # atau membuat dataset_predict_only_raw secara terpisah
                 # Untuk kesederhanaan saat ini, kita asumsikan dataset_test sudah siap untuk diprediksi
                 if 'dataset_test' in locals():
                     dataset_to_predict = dataset_test
                     logger.info("Membuat prediksi pada dataset test (mode predict_only).")
                 else:
                      logger.warning("Dataset test tidak tersedia untuk prediksi dalam mode predict_only.")

             elif config['mode'] in ['initial_train', 'incremental_learn']:
                 # Dalam mode training, prediksi biasanya dilakukan pada dataset test
                 if 'dataset_test' in locals():
                     dataset_to_predict = dataset_test
                     logger.info("Membuat prediksi pada dataset test (mode training).")
                 else:
                     logger.warning("Dataset test tidak tersedia untuk prediksi setelah pelatihan.")

             if dataset_to_predict is not None:
                logger.info(f"Membuat dan menyimpan prediksi ke {predictions_path_full}...")
                # Menggunakan API Keras Model: predict
                # Menggunakan Aritmatika di dalam scaler inverse_transform
                predictions_scaled = model.predict(dataset_to_predict)

                # Pastikan scaler_target tersedia untuk inverse transform
                if 'scaler_target' in locals():
                    predictions_original_scale = scaler_target.inverse_transform(predictions_scaled)

                    # Membuat DataFrame hasil prediksi
                    df_predictions = pd.DataFrame(predictions_original_scale, columns=[f'{col}_Pred' for col in target_cols_present]) # Gunakan kolom target yang benar-benar ada

                    # Opsi: Gabungkan dengan data test asli (membutuhkan penanganan indeks)
                    # Ini bisa rumit dengan windowing. Cara paling aman adalah menyimpan prediksi saja
                    # atau menggabungkan data asli dan prediksi di luar pipeline ini.

                    # Menggunakan Pandas to_csv o r to_json (Menulis)
                    df_predictions.to_csv(predictions_path_full, index=False)
                    logger.info("Prediksi berhasil disimpan.")
                else:
                     logger.warning("Scaler target tidak tersedia. Tidak dapat melakukan inverse transform atau menyimpan prediksi.")

          #  else:
          #       logger.warning("Tidak ada dataset yang ditentukan untuk prediksi.")


        logger.info("Langkah 6 selesai.")

    except Exception as e:
        logger.error(f"Error selama Langkah 6: {e}")


    logger.info("Pipeline QuantAI selesai.")

# --- Eksekusi Pipeline ---

if __name__ == "__main__":
    # Menggunakan argparse untuk membaca path konfigurasi dari command-line
    parser = argparse.ArgumentParser(description='Run QuantAI ML Pipeline.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config_path = args.config

    # Membaca konfigurasi saat script dijalankan
    config = None
    try:
        # Menggunakan Python Standard Library: open
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Konfigurasi berhasil dimuat dari {config_path}")
        # logger.info(f"Konfigurasi: {config}") # Opsi: log seluruh config
    except FileNotFoundError:
        logger.error(f"File konfigurasi tidak ditemukan di {config_path}.")
        # Di GitHub Actions, kita tidak membuat config default jika tidak ditemukan
        # karena path config diharapkan diberikan dengan benar oleh workflow.
        exit(1) # Keluar dengan kode error jika config tidak ditemukan

    except yaml.YAMLError as e:
        logger.error(f"Error mem-parsing file konfigurasi YAML: {e}")
        exit(1) # Keluar dengan kode error jika file config error

    # Jalankan pipeline utama hanya jika config berhasil dimuat
    if config:
        run_pipeline(config)
    else: # <--- Indentasi untuk 'else:' harus sejajar dengan 'if config:'
        logger.error("Tidak ada konfigurasi yang tersedia. Menghentikan eksekusi.") # <--- Indentasi di dalam blok 'else:'
        exit(1) # <--- Indentasi di dalam blok 'else:'

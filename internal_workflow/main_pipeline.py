# -*- coding: utf-8 -*-
"""
Refactored Kode Pipeline QuantAI End-to-End dalam Satu File.

Kode ini mengimplementasikan seluruh alur kerja ML untuk model QuantAI,
dari pemuatan data mentah hingga penyimpanan model terlatih dan hasilnya.
Dirancang untuk dijalankan di lingkungan standar Python,
termasuk runner GitHub Actions.

Fitur Termasuk:
- Membaca konfigurasi dari file YAML (path diberikan sebagai argumen command-line).
- Setup TensorFlow dan Hardware (GPU/Mixed Precision).
- Pemuatan data (CSV/JSON) dan preprocessing awal (termasuk Feature Engineering aritmatika),
  termasuk penanganan kolom 'Date' untuk pengurutan dan output.
- Pembuatan fitur teks deskriptif.
- Scaling data numerik.
- Pembuatan pipeline tf.data yang efisien (windowing, batching, caching, prefetching),
  termasuk penanganan data teks dan target yang selaras.
- Definisi atau pemuatan model QuantAI (Functional API dengan GRU, LayerNorm, Dense,
  potensi input teks).
- Kompilasi model.
- Pelatihan model (mode initial_train atau incremental_learn) dengan callbacks.
- Evaluasi model akhir.
- Penyimpanan model terlatih (.h5), scaler, file kosakata teks, hasil evaluasi, dan prediksi
  (termasuk kolom 'Date' yang selaras).
- Penggunaan API aritmatika, baca/tulis, dan fondasi belajar mandiri/hybrid.

Untuk menjalankan di GitHub Actions:
1. Pastikan kode ini, quantai_config.yaml, requirements.txt ada di repositori.
2. Pastikan data input ada di path yang sesuai di repositori.
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
import json
import argparse
import traceback # Import traceback module untuk logging error detail

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Fungsi Pembantu (Original Helper Functions) ---

def generate_analysis_text(row):
    """
    Menghasilkan string teks deskriptif berdasarkan nilai numerik baris data.
    Menggunakan operasi aritmatika dan perbandingan.
    """
    text = []
    pivot = row.get('Pivot', None)
    close = row.get('Close', None)
    r1 = row.get('R1', None)
    s1 = row.get('S1', None)
    high = row.get('High', None)
    low = row.get('Low', None)

    if pivot is not None and close is not None and pd.notna(pivot) and pd.notna(close):
        if close > pivot:
            text.append(f"Close {close:.2f} di atas Pivot {pivot:.2f}, bullish.")
        elif close < pivot:
            text.append(f"Close {close:.2f} di bawah Pivot {pivot:.2f}, bearish.")
        else:
            text.append(f"Close {close:.2f} di sekitar Pivot {pivot:.2f}, netral.")

    if r1 is not None and high is not None and pd.notna(r1) and pd.notna(high):
        if high > r1:
            text.append(f"High {high:.2f} menembus R1 {r1:.2f}, potensi naik.")

    if s1 is not None and low is not None and pd.notna(s1) and pd.notna(low):
        if low < s1:
            text.append(f"Low {low:.2f} menembus S1 {s1:.2f}, potensi turun.")
    return " ".join(text) if text else "Analisis tidak tersedia."


@tf.function
def process_text_for_model(text_tensor, max_sequence_length, vocabulary_table):
    """
    Memproses tensor string teks menjadi representasi numerik untuk model.
    """
    tokens = tf.strings.split(text_tensor)
    token_ids = vocabulary_table.lookup(tokens)
    current_len = tf.shape(token_ids)[0]
    padded_token_ids = tf.pad(token_ids, [[0, tf.maximum(0, max_sequence_length - current_len)]], constant_values=0)
    padded_token_ids = padded_token_ids[:max_sequence_length]
    return padded_token_ids


def build_quantai_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size=None):
    """
    Membangun arsitektur model QuantAI menggunakan Functional API.
    Mendukung input numerik dan teks.
    """
    numeric_input = tf.keras.layers.Input(shape=numeric_input_shape, name='numeric_input')
    x_numeric = numeric_input
    
    gru_units = config['model']['architecture'].get('gru_units', 64)
    dropout_rate = config['model']['architecture'].get('dropout_rate', 0.2)
    embedding_dim = config['model']['architecture'].get('embedding_dim', 32)

    x_numeric = tf.keras.layers.GRU(gru_units, return_sequences=True)(x_numeric)
    x_numeric = tf.keras.layers.LayerNormalization()(x_numeric)
    x_numeric = tf.keras.layers.Dropout(dropout_rate)(x_numeric)

    if config['model']['architecture'].get('use_conv1d', False):
        conv1d_filters = config['model']['architecture'].get('conv1d_filters', 32)
        conv1d_kernel_size = config['model']['architecture'].get('conv1d_kernel_size', 3)
        conv1d_dilation_rate = config['model']['architecture'].get('conv1d_dilation_rate', 1)
        x_numeric = tf.keras.layers.Conv1D(
            filters=conv1d_filters,
            kernel_size=conv1d_kernel_size,
            activation='relu',
            padding='causal',
            dilation_rate=conv1d_dilation_rate
        )(x_numeric)
        x_numeric = tf.keras.layers.LayerNormalization()(x_numeric)
        x_numeric = tf.keras.layers.Dropout(dropout_rate)(x_numeric)
    
    x_numeric = tf.keras.layers.GlobalAveragePooling1D()(x_numeric)

    use_text_input = config.get('use_text_input', False)
    combined_input = x_numeric
    inputs = [numeric_input]

    if use_text_input and text_input_shape is not None and vocabulary_size is not None and vocabulary_size > 1:
        text_input = tf.keras.layers.Input(shape=text_input_shape, name='text_input', dtype=tf.int64)
        x_text = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            mask_zero=True
        )(text_input)
        x_text = tf.keras.layers.GlobalAveragePooling1D()(x_text) # Pool across max_text_sequence_length
        x_text = tf.keras.layers.GRU(gru_units // 2)(x_text) # Process sequence across window_size
        x_text = tf.keras.layers.LayerNormalization()(x_text)
        x_text = tf.keras.layers.Dropout(dropout_rate)(x_text)
        
        combined_input = tf.keras.layers.Concatenate()([x_numeric, x_text])
        inputs.append(text_input)
    elif use_text_input:
        logger.warning("Fitur teks diaktifkan tetapi tidak dapat membuat jalur teks model. Menggunakan hanya jalur numerik.")

    output_layer = tf.keras.layers.Dense(num_target_features, name='output')(combined_input)
    model = tf.keras.Model(inputs=inputs, outputs=output_layer)
    return model

# --- Pipeline Steps ---

def setup_hardware_and_seeds(config):
    """Langkah 0.5: Setup TensorFlow Hardware dan Global Seeds."""
    logger.info("Langkah 0.5: Mengkonfigurasi TensorFlow, Hardware, dan Seeds...")
    try:
        physical_devices_gpu = tf.config.list_physical_devices('GPU')
        if physical_devices_gpu:
            logger.info(f"Ditemukan GPU: {len(physical_devices_gpu)}")
            tf.config.set_visible_devices(physical_devices_gpu, 'GPU')
            for gpu in physical_devices_gpu:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth diaktifkan untuk GPU.")
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision (mixed_float16) diaktifkan.")
            except Exception as mp_e:
                logger.warning(f"Gagal mengaktifkan mixed precision: {mp_e}. Melanjutkan tanpa mixed precision.")
        else:
            logger.warning("Tidak ada GPU yang terdeteksi. Menggunakan CPU.")
            tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'), 'CPU')

        seed = config['seed']
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Seed ({seed}) berhasil disetel. Setup hardware dan seed selesai.")
        return True
    except Exception as e:
        logger.error(f"Gagal mengkonfigurasi hardware TensorFlow atau seed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def load_and_preprocess_data(config):
    """Langkah 1: Pemuatan Data & Preprocessing Awal."""
    logger.info("Langkah 1: Memuat data dan preprocessing awal...")
    try:
        data_path = config['data']['raw_path']
        date_column_name = config['data'].get('date_column', 'Date')

        if not os.path.exists(data_path):
            logger.error(f"File data tidak ditemukan di {os.path.abspath(data_path)}.")
            return None, None, None, None, None, None # df, dates, features_numeric, features_text, targets, test_start_idx

        # Load data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, sep=config['data'].get('delimiter', ','))
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Format data tidak didukung. Gunakan .csv atau .json.")
        logger.info(f"Data mentah dimuat dari {data_path}. Jumlah baris: {len(df)}")

        # Date handling and sorting
        original_dates_for_alignment = None
        if date_column_name in df.columns:
            df[date_column_name] = pd.to_datetime(df[date_column_name])
            df.sort_values(by=date_column_name, inplace=True)
            original_dates_for_alignment = df[date_column_name].copy()
            logger.info(f"Data diurutkan berdasarkan '{date_column_name}'.")
        else:
            logger.warning(f"Kolom '{date_column_name}' tidak ditemukan.")
            date_column_name = None # Ensure it's None if not found

        # Initial NaN Cleaning
        cols_to_check_nan = list(config['data']['feature_cols_numeric'])
        ohlcv_cols_basic = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_cols_basic:
            if col in df.columns and col not in cols_to_check_nan:
                 cols_to_check_nan.append(col)
        
        if config['data'].get('include_date_in_nan_check', False) and date_column_name:
            cols_to_check_nan.append(date_column_name)

        cols_to_check_nan = [col for col in cols_to_check_nan if col in df.columns] # Filter to existing columns
        if cols_to_check_nan:
            initial_rows = len(df)
            df.dropna(subset=cols_to_check_nan, inplace=True)
            if len(df) < initial_rows:
                logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di {cols_to_check_nan}.")
        else:
            logger.warning("Tidak ada kolom valid untuk pengecekan NaN awal.")


        # Feature Engineering (Pivot Points)
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in df.columns for col in ohlc_cols):
            df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['R1'] = 2 * df['Pivot'] - df['Low']
            df['S1'] = 2 * df['Pivot'] - df['High']
            # ... (add R2/S2, R3/S3 as in original)
            df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
            df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
            df['R3'] = df['Pivot'] + 2 * (df['High'] - df['Low'])
            df['S3'] = df['Pivot'] - 2 * (df['High'] - df['Low'])
            logger.info("Perhitungan Pivot Points selesai.")
        else:
            logger.warning("Kolom OHLC tidak lengkap. Tidak dapat menghitung Pivot Points.")

        # Target Creation
        window_size = config['parameter_windowing']['window_size']
        target_cols = ['High_Next', 'Low_Next', 'Close_Next']
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            df['High_Next'] = df['High'].shift(-window_size)
            df['Low_Next'] = df['Low'].shift(-window_size)
            df['Close_Next'] = df['Close'].shift(-window_size)
            
            initial_rows = len(df)
            df.dropna(subset=target_cols, inplace=True) # Drop NaNs created by shift
            if len(df) < initial_rows:
                logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di target setelah shift.")
            logger.info("Target HLC selanjutnya dibuat.")
        else:
            logger.error("Kolom High, Low, Close tidak tersedia untuk membuat target.")
            return None, None, None, None, None, None

        # Text Feature Generation
        feature_cols_text_names = []
        if config.get('use_text_input', False):
            # Ensure necessary columns for generate_analysis_text exist
            cols_needed_for_text = ['High', 'Low', 'Close'] + [col for col in df.columns if col.startswith(('Piv', 'R', 'S'))]
            if all(col in df.columns for col in cols_needed_for_text):
                df['Analysis_Text'] = df.apply(generate_analysis_text, axis=1)
                feature_cols_text_names = ['Analysis_Text']
                if not df.empty: logger.info(f"Contoh teks analisis: '{df['Analysis_Text'].iloc[0]}'")
            else:
                logger.warning("Kolom untuk generate_analysis_text tidak lengkap. Fitur teks tidak dibuat.")
        
        # Identify Feature and Target Columns
        indicator_cols = [col for col in df.columns if col.startswith(('Piv', 'R', 'S'))]
        feature_cols_numeric_names = list(config['data']['feature_cols_numeric'])
        for col in indicator_cols:
            if col not in feature_cols_numeric_names and col in df.columns:
                feature_cols_numeric_names.append(col)
        
        if date_column_name and date_column_name in feature_cols_numeric_names:
            feature_cols_numeric_names.remove(date_column_name)

        feature_cols_numeric_names = [col for col in feature_cols_numeric_names if col in df.columns] # Ensure all exist

        if df.empty:
            logger.error("DataFrame kosong setelah preprocessing.")
            return None, None, None, None, None, None

        all_feature_target_cols = feature_cols_numeric_names + feature_cols_text_names + target_cols
        missing_final_cols = [col for col in all_feature_target_cols if col not in df.columns]
        if missing_final_cols:
            logger.error(f"Kolom fitur/target akhir tidak ditemukan: {missing_final_cols}")
            return None, None, None, None, None, None

        df_processed = df[all_feature_target_cols].copy()

        # Data Splitting (indices based on processed DataFrame)
        total_samples = len(df_processed)
        train_split_ratio = config['data'].get('train_split', 0.8)
        val_split_ratio = config['data'].get('val_split', 0.1)

        if not (0 < train_split_ratio < 1 and 0 < val_split_ratio < 1 and (train_split_ratio + val_split_ratio) < 1):
            logger.error("Split ratio tidak valid.")
            return None, None, None, None, None, None

        train_size = int(total_samples * train_split_ratio)
        val_size = int(total_samples * val_split_ratio)
        test_split_start_idx = train_size + val_size
        
        if train_size == 0 or val_size == 0 or (total_samples - test_split_start_idx) == 0:
            logger.error("Ukuran set data tidak memadai setelah split.")
            return None, None, None, None, None, None

        # Convert to NumPy
        data_numeric_np = df_processed[feature_cols_numeric_names].values.astype(np.float32)
        data_text_np = df_processed[feature_cols_text_names].values.flatten().astype(str) if feature_cols_text_names else np.array([], dtype=str)
        data_target_np = df_processed[target_cols].values.astype(np.float32)

        logger.info(f"Final data shapes: Numeric {data_numeric_np.shape}, Text {data_text_np.shape}, Target {data_target_np.shape}")
        
        # Return data splits and supporting info
        # Slicing numpy arrays for splits
        input_train_numeric = data_numeric_np[:train_size]
        input_val_numeric = data_numeric_np[train_size:test_split_start_idx]
        input_test_numeric = data_numeric_np[test_split_start_idx:]

        input_train_text = data_text_np[:train_size] if feature_cols_text_names else np.array([], dtype=str)
        input_val_text = data_text_np[train_size:test_split_start_idx] if feature_cols_text_names else np.array([], dtype=str)
        input_test_text = data_text_np[test_split_start_idx:] if feature_cols_text_names else np.array([], dtype=str)

        target_train = data_target_np[:train_size]
        target_val = data_target_np[train_size:test_split_start_idx]
        target_test = data_target_np[test_split_start_idx:]

        data_splits = {
            "train_num": input_train_numeric, "val_num": input_val_numeric, "test_num": input_test_numeric,
            "train_text": input_train_text, "val_text": input_val_text, "test_text": input_test_text,
            "train_target": target_train, "val_target": target_val, "test_target": target_test
        }
        
        logger.info(f"Train set size: {len(input_train_numeric)}, Val set size: {len(input_val_numeric)}, Test set size: {len(input_test_numeric)}")

        return (data_splits, original_dates_for_alignment, date_column_name, 
                feature_cols_numeric_names, feature_cols_text_names, target_cols, test_split_start_idx)

    except Exception as e:
        logger.error(f"Error selama Langkah 1 (load_and_preprocess_data): {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None, None, None, None


def scale_data_and_create_tf_datasets(config, data_splits, feature_cols_text_names):
    """Langkah 2: Scaling Data & Pembuatan Pipeline tf.data."""
    logger.info("Langkah 2: Scaling data dan membuat pipeline tf.data...")
    try:
        scaler_input = None
        scaler_target = None
        vocabulary_table = None
        vocabulary_size = 0
        max_text_sequence_length = config['data'].get('max_text_sequence_length', 20)

        # Scale numeric data
        if data_splits["train_num"].size > 0:
            scaler_input = MinMaxScaler()
            scaler_target = MinMaxScaler()

            scaled_input_train = scaler_input.fit_transform(data_splits["train_num"])
            scaled_target_train = scaler_target.fit_transform(data_splits["train_target"])
            scaled_input_val = scaler_input.transform(data_splits["val_num"])
            scaled_target_val = scaler_target.transform(data_splits["val_target"])
            scaled_input_test = scaler_input.transform(data_splits["test_num"])
            # target_test remains unscaled for evaluation, but predictions will be inverse_transformed using scaler_target
            logger.info("Data numerik berhasil diskalakan.")
        else: # Handle cases with no numeric data or all zeros/NaNs
            logger.warning("Data numerik pelatihan tidak tersedia atau kosong. Scaling dilewati.")
            scaled_input_train = data_splits["train_num"]
            scaled_target_train = data_splits["train_target"]
            scaled_input_val = data_splits["val_num"]
            scaled_target_val = data_splits["val_target"]
            scaled_input_test = data_splits["test_num"]

        # Text vocabulary
        if config.get('use_text_input', False) and feature_cols_text_names and data_splits["train_text"].size > 0:
            logger.info("Membuat kosakata dari data pelatihan teks...")
            all_train_tokens_ragged = tf.strings.split(tf.constant(data_splits["train_text"].tolist(), dtype=tf.string))
            unique_tokens_np = np.unique(all_train_tokens_ragged.values.numpy()[all_train_tokens_ragged.values.numpy() != b''])
            
            if unique_tokens_np.size > 0:
                keys = tf.constant(unique_tokens_np, dtype=tf.string)
                values = tf.range(1, tf.size(keys) + 1, dtype=tf.int64) # Reserve 0 for padding
                init = tf.lookup.KeyValueTensorInitializer(keys, values)
                vocabulary_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=1)
                vocabulary_size = vocabulary_table.size().numpy() + 1 # +1 for padding 0
                logger.info(f"Ukuran kosakata: {vocabulary_size}. Contoh: {unique_tokens_np[:5]}")
            else:
                logger.warning("Tidak ada token unik ditemukan. Kosakata tidak dibuat.")
                vocabulary_size = 2 # PAD=0, OOV=1
        elif config.get('use_text_input', False):
            logger.warning("Fitur teks diaktifkan, tetapi data teks pelatihan kosong atau kolom teks tidak ada.")
            vocabulary_size = 2 # PAD=0, OOV=1
        
        # Create tf.data.Dataset
        datasets = {}
        for split_name in ["train", "val", "test"]:
            current_num_data = locals().get(f'scaled_input_{split_name}', data_splits[f"{split_name}_num"]) # Use scaled for train/val, original for test num before windowing
            current_text_data = data_splits[f"{split_name}_text"]
            current_target_data = locals().get(f'scaled_target_{split_name}', data_splits[f"{split_name}_target"]) # Use scaled for train/val targets

            if current_num_data.shape[0] == 0 and current_text_data.shape[0] == 0:
                logger.warning(f"Tidak ada data untuk membuat dataset {split_name}.")
                datasets[split_name + "_raw"] = None
                continue

            if feature_cols_text_names and config.get('use_text_input', False):
                datasets[split_name + "_raw"] = tf.data.Dataset.from_tensor_slices(((current_num_data, current_text_data), current_target_data))
            else:
                datasets[split_name + "_raw"] = tf.data.Dataset.from_tensor_slices((current_num_data, current_target_data))

        # Windowing function
        @tf.function
        def process_window(input_elements, target_window, use_text, max_len, vocab_tbl):
            numeric_window = input_elements[0] if use_text else input_elements
            text_window = input_elements[1] if use_text else None
            final_target = target_window[-1] # Target is the last element of the target window

            if use_text and text_window is not None and vocab_tbl is not None and max_len > 0:
                processed_text_window = tf.map_fn(
                    lambda txt: process_text_for_model(txt, max_len, vocab_tbl),
                    text_window, fn_output_signature=tf.int64
                )
                return (numeric_window, processed_text_window), final_target
            return numeric_window, final_target

        map_params = {
            'use_text': config.get('use_text_input', False) and feature_cols_text_names and vocabulary_size > 1,
            'max_len': max_text_sequence_length,
            'vocab_tbl': vocabulary_table
        }
        window_size = config['parameter_windowing']['window_size']
        window_shift = config['parameter_windowing'].get('window_shift', 1)
        batch_size = config['training'].get('batch_size', 32)

        for split_name in ["train", "val", "test"]:
            raw_ds = datasets.get(split_name + "_raw")
            if raw_ds is None:
                datasets[split_name] = None
                continue
            
            windowed_ds = raw_ds.window(size=window_size, shift=window_shift, drop_remainder=True)
            # flat_map to batch elements within each window
            # For train/val, input_elements is a tuple (numeric_window, text_window) or just numeric_window
            # For train/val, target_window is just the target_window
            # For test, target_window is target_test (unscaled)
            # The map function expects (input_elements, target_window)
            
            # Adjusting the flat_map to handle the structure correctly
            if map_params['use_text']:
                 # When text is used, raw_ds elements are ((num, txt), target)
                 # window.batch(window_size) will create:
                 # ((batched_num, batched_txt), batched_target)
                processed_ds = windowed_ds.flat_map(
                    lambda num_txt_window, target_window_ds: tf.data.Dataset.zip(
                        (num_txt_window[0].batch(window_size), num_txt_window[1].batch(window_size)), 
                        target_window_ds.batch(window_size)
                    )
                )
                # Now map needs to unpack num_txt_window correctly
                processed_ds = processed_ds.map(
                    lambda num_txt_batched, target_batched: process_window(num_txt_batched, target_batched, **map_params),
                    num_parallel_calls=tf.data.AUTOTUNE
                )

            else: # Only numeric data
                 # raw_ds elements are (num, target)
                 # window.batch(window_size) will create:
                 # (batched_num, batched_target)
                processed_ds = windowed_ds.flat_map(
                     lambda num_window_ds, target_window_ds: tf.data.Dataset.zip(
                         (num_window_ds.batch(window_size), target_window_ds.batch(window_size))
                     )
                )
                processed_ds = processed_ds.map(
                    lambda num_batched, target_batched: process_window(num_batched, target_batched, **map_params),
                    num_parallel_calls=tf.data.AUTOTUNE
                )


            if split_name == "train":
                processed_ds = processed_ds.shuffle(config['data'].get('shuffle_buffer_size', 1000))
            
            datasets[split_name] = processed_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE if split_name != "test" else tf.data.AUTOTUNE) # Caching for train/val
            logger.info(f"Pipeline {split_name} tf.data: {datasets[split_name].element_spec if datasets[split_name] else 'Not created'}")

        return datasets, scaler_input, scaler_target, vocabulary_table, vocabulary_size, unique_tokens_np if 'unique_tokens_np' in locals() else None

    except Exception as e:
        logger.error(f"Error selama Langkah 2 (scale_data_and_create_tf_datasets): {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None, None, None


def define_or_load_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size):
    """Langkah 3: Definisi & Kompilasi Model."""
    logger.info("Langkah 3: Mendefinisikan atau memuat model...")
    model = None
    try:
        if config['mode'] == 'initial_train':
            model = build_quantai_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size)
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['training'].get('learning_rate', 0.001))
            model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsoluteError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
            logger.info("Model baru berhasil didefinisikan dan dikompilasi.")
            model.summary(print_fn=logger.info)
        
        elif config['mode'] in ['incremental_learn', 'predict_only']:
            load_path = config['model'].get('load_path')
            if not load_path or not os.path.exists(load_path):
                logger.error(f"Model path {load_path} tidak ditemukan atau tidak dikonfigurasi.")
                return None
            
            model = tf.keras.models.load_model(load_path) # Assumes H5 format from original
            logger.info(f"Model berhasil dimuat dari {load_path}.")
            
            if config['mode'] == 'incremental_learn':
                optimizer = tf.keras.optimizers.Adam(learning_rate=config['training'].get('learning_rate', 0.001)) # Recompile for training
                model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsoluteError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
                logger.info("Model dikompilasi ulang untuk incremental learning.")
        else:
            logger.error(f"Mode operasional tidak valid: {config['mode']}")
            return None
        
        return model

    except Exception as e:
        logger.error(f"Error selama Langkah 3 (define_or_load_model): {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def train_model_pipeline(config, model, datasets, model_save_path_full):
    """Langkah 4: Pelatihan Model."""
    if not (config['mode'] in ['initial_train', 'incremental_learn'] and model and datasets.get('train') and datasets.get('val')):
        logger.info("Melewatkan Langkah 4: Pelatihan model tidak diperlukan atau data/model tidak siap.")
        return model # Return original model if not trained

    logger.info(f"Langkah 4: Memulai pelatihan model dalam mode {config['mode']}...")
    try:
        output_dir = config['output']['base_dir']
        tensorboard_log_dir_full = os.path.join(output_dir, config['output'].get('tensorboard_log_dir', 'logs/tensorboard'))
        tf.io.gfile.makedirs(os.path.dirname(model_save_path_full)) # Ensure save directory exists
        tf.io.gfile.makedirs(tensorboard_log_dir_full)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training'].get('early_stopping_patience', 10), restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path_full, monitor='val_loss', save_best_only=True, save_format='h5'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['training'].get('lr_reduce_factor', 0.5), patience=config['training'].get('lr_reduce_patience', 5)),
            tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir_full)
        ]
        epochs = config['training'].get('epochs', 100) if config['mode'] == 'initial_train' else config['training'].get('incremental_epochs', 10)

        model.fit(datasets['train'], epochs=epochs, validation_data=datasets['val'], callbacks=callbacks)
        logger.info("Pelatihan selesai.")
        
        # ModelCheckpoint saves the best model, so we load it back if training occurred.
        # restore_best_weights in EarlyStopping should handle this for the in-memory model,
        # but loading from disk ensures we have the one saved by ModelCheckpoint.
        if os.path.exists(model_save_path_full):
            logger.info(f"Memuat model terbaik dari {model_save_path_full} setelah pelatihan.")
            model = tf.keras.models.load_model(model_save_path_full) # Reload best model
        else:
            logger.warning(f"Model file {model_save_path_full} tidak ditemukan setelah pelatihan. Menggunakan model terakhir di memori.")

        return model

    except Exception as e:
        logger.error(f"Error selama Langkah 4 (train_model_pipeline): {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return model # Return original model in case of error during training


def evaluate_model_pipeline(model, datasets):
    """Langkah 5: Evaluasi Model Akhir."""
    if not (model and datasets.get('test') and hasattr(model, 'evaluate')):
        logger.info("Melewatkan Langkah 5: Evaluasi model tidak diperlukan atau data/model tidak siap.")
        return None

    logger.info("Langkah 5: Mengevaluasi model akhir...")
    try:
        eval_results = model.evaluate(datasets['test'])
        if hasattr(model, 'metrics_names'):
            results_dict = dict(zip(model.metrics_names, eval_results))
            logger.info(f"Hasil Evaluasi Akhir: {results_dict}")
            return results_dict
        logger.info(f"Hasil Evaluasi Akhir (raw): {eval_results}")
        return eval_results
    except Exception as e:
        logger.error(f"Error selama Langkah 5 (evaluate_model_pipeline): {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def save_artifacts_and_predictions(config, model, scaler_input, scaler_target, vocabulary_tokens, eval_results,
                                   datasets, original_dates, date_col_name, target_cols,
                                   test_split_start_idx, window_size, model_save_path_full):
    """Langkah 6: Penyimpanan Aset & Hasil."""
    logger.info("Langkah 6: Menyimpan aset dan hasil...")
    try:
        output_dir = config['output']['base_dir']
        
        # Model (already saved by ModelCheckpoint during training if applicable)
        if config['mode'] in ['initial_train', 'incremental_learn']:
            if os.path.exists(model_save_path_full):
                 logger.info(f"Model terbaik sudah tersimpan di: {model_save_path_full}")
            else:
                 # This case implies training was skipped or ModelCheckpoint failed to save.
                 # If model needs to be saved explicitly outside training (e.g. predict_only with loaded model for resave)
                 # an explicit model.save() would be needed here. For now, assume training handles it.
                 logger.warning(f"Model file ({model_save_path_full}) tidak ditemukan. Model mungkin tidak tersimpan jika pelatihan dilewati/gagal.")


        # Scalers
        scaler_save_dir = os.path.join(output_dir, config['output'].get('scaler_subdir', 'scalers'))
        tf.io.gfile.makedirs(scaler_save_dir)
        if scaler_input: joblib.dump(scaler_input, os.path.join(scaler_save_dir, 'scaler_input.pkl'))
        if scaler_target: joblib.dump(scaler_target, os.path.join(scaler_save_dir, 'scaler_target.pkl'))
        logger.info(f"Scaler disimpan ke {scaler_save_dir}.")

        # Vocabulary
        if config.get('use_text_input', False) and vocabulary_tokens is not None and vocabulary_tokens.size > 0:
            vocab_file_path = os.path.join(output_dir, config['output'].get('vocabulary_file', os.path.join(config['output'].get('scaler_subdir', 'scalers'), 'vocabulary.txt')))
            tf.io.gfile.makedirs(os.path.dirname(vocab_file_path))
            with open(vocab_file_path, 'w', encoding='utf-8') as f:
                for token in vocabulary_tokens:
                    f.write(token.decode('utf-8') + '\n')
            logger.info(f"Kosakata disimpan ke {vocab_file_path}.")

        # Evaluation Results
        if eval_results:
            eval_file_path = os.path.join(output_dir, config['output'].get('eval_results_file', 'evaluation_metrics/eval_results.json'))
            tf.io.gfile.makedirs(os.path.dirname(eval_file_path))
            with open(eval_file_path, 'w') as f:
                # Ensure eval_results is JSON serializable (it should be if it's a dict from Keras)
                serializable_results = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in eval_results.items()} if isinstance(eval_results, dict) else eval_results
                json.dump(serializable_results, f, indent=4)
            logger.info(f"Hasil evaluasi disimpan ke {eval_file_path}.")

        # Predictions
        if config['output'].get('save_predictions', False) and model and datasets.get('test') and scaler_target:
            logger.info("Membuat dan menyimpan prediksi...")
            predictions_scaled = model.predict(datasets['test'])
            predictions_original = scaler_target.inverse_transform(predictions_scaled)
            
            df_predictions = pd.DataFrame(predictions_original, columns=[f'{col}_Pred' for col in target_cols])

            if original_dates is not None and date_col_name and test_split_start_idx is not None and window_size is not None:
                # Align dates (simplified logic from original, ensure correctness for your windowing)
                num_predictions = len(df_predictions)
                # The first prediction corresponds to the data at test_split_start_idx + window_size -1 in original_dates index
                # This needs careful alignment based on how windowing affects indices.
                # The original code had: date_indices_for_predictions_in_original = np.arange(0, num_predictions) * window_shift + (test_split_start_idx_processed_df + window_size)
                # Assuming window_shift = 1 for this example, and test_split_start_idx refers to the start of the *test data before windowing*.
                # The first *input window* for testing ends at original_df_index = test_split_start_idx + window_size -1. The target corresponds to this.
                
                # A more robust way if `dataset_test_raw` was available before windowing and batching:
                # test_dates = original_dates.iloc[test_split_start_idx + window_size -1 : test_split_start_idx + window_size -1 + num_predictions]
                # For now, let's try to replicate based on available info, assuming shift=1
                
                # The `target_test` data (which `predictions_original` corresponds to) starts effectively from
                # `test_split_start_idx + window_size -1` index of the original `df` (before targets were dropped)
                # `original_dates_for_alignment` is from the df *before* target NaNs were dropped.
                # `test_split_start_idx` is index in df_processed (after target NaNs dropped).
                # This alignment is tricky and needs the exact same indexing logic as during data prep.
                # Let's use a simplified approach, assuming date alignment should match the target_test set.
                
                # Assuming test_split_start_idx is for df_processed (after target creation and NaN drop)
                # and original_dates_for_alignment is the full date column from the initial load.
                # The dates for predictions should correspond to the dates of the *targets* in the test set.
                # If target_test came from df_processed[test_split_start_idx:], then dates for predictions should be from
                # original_dates_for_alignment.iloc corresponding to those rows.
                # This requires careful tracking of indices through all preprocessing.
                # For simplicity, this example will assume the number of predictions matches a slice of original_dates.
                # THIS PART NEEDS CAREFUL VALIDATION BASED ON YOUR EXACT DATA PREP.
                
                # A common pattern is to extract the dates relevant to the test targets *before* windowing the test dataset.
                # Let's assume `test_split_start_idx` is the start of test data in the *processed* dataframe (df_processed).
                # The target for the first window corresponds to the data point at index `test_split_start_idx + window_size -1` in `df_processed`.
                # The date for this target in `original_dates_for_alignment` needs to be found.
                
                # Simplified alignment:
                # If `original_dates_for_alignment` was sliced IDENTICALLY to how `df_processed` was created (same NaN drops)
                # then `original_dates_for_alignment.iloc[test_split_start_idx + window_size -1 : test_split_start_idx + window_size - 1 + num_predictions]`
                # However, `original_dates_for_alignment` is from the *very beginning*.
                # The original script's alignment logic for predictions was:
                # date_indices_for_predictions_in_original = np.arange(0, num_predictions) * window_shift + (test_split_start_idx_processed_df + window_size)
                # valid_date_indices_in_original = date_indices_for_predictions_in_original[date_indices_for_predictions_in_original < len(original_dates_for_alignment)]
                # predicted_dates = original_dates_for_alignment.iloc[valid_date_indices_in_original].reset_index(drop=True)

                # Let's try to use the original logic directly, assuming window_shift=1 as it's not passed here.
                window_shift_pred = config['parameter_windowing'].get('window_shift', 1) # Get it from config
                date_indices = np.arange(0, num_predictions) * window_shift_pred + (test_split_start_idx + window_size -1) # -1 because target is at end of window
                
                # Ensure indices are within bounds of original_dates
                valid_indices = date_indices[date_indices < len(original_dates)]
                
                if len(valid_indices) == num_predictions:
                    predicted_dates = original_dates.iloc[valid_indices].reset_index(drop=True)
                    df_predictions.insert(0, date_col_name, predicted_dates)
                    logger.info(f"Kolom '{date_col_name}' ditambahkan ke prediksi.")
                else:
                    logger.warning(f"Gagal align tanggal untuk prediksi. Jumlah tanggal valid ({len(valid_indices)}) tidak cocok dengan jumlah prediksi ({num_predictions}).")
                    logger.warning(f"Indeks tanggal yang dicoba: {date_indices[:10]}...")
                    logger.warning(f"Panjang original_dates: {len(original_dates)}")


            pred_file_path = os.path.join(output_dir, config['output'].get('predictions_file', 'predictions/predictions.csv'))
            tf.io.gfile.makedirs(os.path.dirname(pred_file_path))
            df_predictions.to_csv(pred_file_path, index=False)
            logger.info(f"Prediksi disimpan ke {pred_file_path}.")
        elif config['output'].get('save_predictions', False):
            logger.warning("Tidak dapat menyimpan prediksi: model, dataset test, atau scaler target tidak tersedia.")

        logger.info("Langkah 6 (penyimpanan) selesai.")
        return True

    except Exception as e:
        logger.error(f"Error selama Langkah 6 (save_artifacts_and_predictions): {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# --- Main Pipeline Orchestration ---
def run_full_pipeline(config):
    """
    Orchestrates the execution of the entire ML pipeline.
    """
    logger.info("Memulai pipeline QuantAI (versi refactored)...")

    if not setup_hardware_and_seeds(config):
        logger.error("Setup hardware gagal. Pipeline dihentikan.")
        return

    # Langkah 1: Load and Preprocess Data
    (data_splits, original_dates, date_col_name, 
     feature_cols_numeric_names, feature_cols_text_names, target_cols, 
     test_split_start_idx) = load_and_preprocess_data(config)
    
    if data_splits is None:
        logger.error("Pemuatan dan preprocessing data gagal. Pipeline dihentikan.")
        return

    # Langkah 2: Scale Data and Create tf.data Datasets
    (datasets, scaler_input, scaler_target, vocabulary_table, 
     vocabulary_size, vocabulary_tokens) = scale_data_and_create_tf_datasets(config, data_splits, feature_cols_text_names)

    if datasets is None: # Check if dataset creation failed critically
        logger.error("Pembuatan dataset tf.data gagal. Pipeline dihentikan.")
        return
    
    # Determine input shapes for the model
    window_size = config['parameter_windowing']['window_size']
    numeric_input_shape = (window_size, len(feature_cols_numeric_names)) if feature_cols_numeric_names else (window_size, 0)
    max_text_sequence_length = config['data'].get('max_text_sequence_length', 20)
    text_input_shape = (window_size, max_text_sequence_length) if config.get('use_text_input', False) and feature_cols_text_names else None
    num_target_features = len(target_cols)

    # Langkah 3: Define or Load Model
    model = define_or_load_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size)
    if model is None:
        logger.error("Definisi atau pemuatan model gagal. Pipeline dihentikan.")
        return

    # Path for saving the model (used in training and saving artifacts)
    output_dir_main = config['output']['base_dir']
    model_save_file_config_main = config['output'].get('model_save_file', 'saved_model/best_model.h5')
    model_save_path_full_main = os.path.join(output_dir_main, model_save_file_config_main)


    # Langkah 4: Train Model
    model = train_model_pipeline(config, model, datasets, model_save_path_full_main)
    # Model is updated in-place or reloaded if training occurred

    # Langkah 5: Evaluate Model
    eval_results = evaluate_model_pipeline(model, datasets)
    # eval_results will be None if evaluation is skipped or fails

    # Langkah 6: Save Artifacts and Predictions
    save_artifacts_and_predictions(config, model, scaler_input, scaler_target, vocabulary_tokens, eval_results,
                                   datasets, original_dates, date_col_name, target_cols,
                                   test_split_start_idx, window_size, model_save_path_full_main)
    
    logger.info("Pipeline QuantAI (versi refactored) selesai.")


# --- Eksekusi Pipeline ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Refactored QuantAI ML Pipeline.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config = None
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Konfigurasi berhasil dimuat dari {os.path.abspath(args.config)}")
    except FileNotFoundError:
        logger.error(f"File konfigurasi tidak ditemukan di {os.path.abspath(args.config)}.")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing file YAML: {e}")
        exit(1)

    if config:
        # Simplified validation from original, ensure it's robust for your needs
        essential_keys = ['data', 'model', 'training', 'output', 'parameter_windowing', 'seed', 'mode']
        if any(key not in config for key in essential_keys):
            logger.error(f"Kunci konfigurasi esensial hilang. Perlu: {essential_keys}")
            exit(1)
        
        # Add default output paths if missing (similar to original script)
        output_cfg = config['output']
        base_dir = output_cfg.get('base_dir', 'outputs') # Default base_dir
        output_cfg['base_dir'] = base_dir # Ensure it's in config for functions
        
        default_paths = {
            'model_save_file': os.path.join('saved_model', 'best_model.h5'),
            'scaler_subdir': 'scalers',
            'tensorboard_log_dir': os.path.join('logs', 'tensorboard'),
            'eval_results_file': os.path.join('evaluation_metrics', 'eval_results.json'),
            'predictions_file': os.path.join('predictions', 'predictions.csv'),
        }
        if config.get('use_text_input', False):
            default_paths['vocabulary_file'] = os.path.join(output_cfg.get('scaler_subdir', 'scalers'), 'vocabulary.txt')

        for key, default_suffix in default_paths.items():
            if key not in output_cfg:
                output_cfg[key] = os.path.join(base_dir, default_suffix) # Original combined base_dir later, doing it here for consistency
                logger.info(f"Menambahkan path default untuk output.{key}: {output_cfg[key]}")
            elif not os.path.isabs(output_cfg[key]) and key not in ['scaler_subdir', 'tensorboard_log_dir']: # if relative, make it relative to base_dir
                 # Special handling for subdirs vs full file paths needs care
                 # The original logic for path construction was a bit mixed.
                 # For simplicity, if a path is provided in config and it's not absolute, assume it's a suffix to base_dir
                 # This might need adjustment based on exact original intent
                 if key not in ['scaler_subdir']: # scaler_subdir is already a subdir name
                    output_cfg[key] = os.path.join(base_dir, output_cfg[key])


        # Mode validation
        if config['mode'] in ['incremental_learn', 'predict_only'] and not config['model'].get('load_path'):
            logger.error(f"Mode '{config['mode']}' memerlukan 'model.load_path'.")
            exit(1)

        run_full_pipeline(config)
    else:
        logger.error("Konfigurasi tidak termuat.")
        exit(1)

# (Previous imports and helper functions remain the same)

# --- Main Pipeline Function ---

def run_pipeline(config):
    """
    Menjalankan seluruh pipeline ML berdasarkan konfigurasi.
    Ini adalah inti dari script satu file.
    """
    logger.info("Memulai pipeline QuantAI...")

    # --- Langkah 0.5: Setup Hardware ---
    # (Hardware setup code remains the same)
    try:
        logger.info("Mengkonfigurasi TensorFlow dan Hardware...")
        physical_devices_gpu = tf.config.list_physical_devices('GPU')
        if physical_devices_gpu:
            logger.info(f"Ditemukan GPU: {len(physical_devices_gpu)}")
            tf.config.set_visible_devices(physical_devices_gpu, 'GPU')
            for gpu in physical_devices_gpu:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth diaktifkan untuk GPU.")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision (mixed_float16) diaktifkan.")
        else:
            logger.warning("Tidak ada GPU yang terdeteksi. Menggunakan CPU.")
            physical_devices_cpu = tf.config.list_physical_devices('CPU')
            tf.config.set_visible_devices(physical_devices_cpu, 'CPU')

        logger.info(f"Menyetel seed: {config['seed']}")
        np.random.seed(config['seed'])
        tf.random.set_seed(config['seed'])
        logger.info("Setup hardware dan seed selesai.")

    except Exception as e:
        logger.warning(f"Gagal mengkonfigurasi hardware TensorFlow: {e}")


    # --- Langkah 1: Pemuatan Data & Preprocessing Awal ---
    logger.info("Langkah 1: Memuat data dan preprocessing awal...")
    df = None # Inisialisasi df
    original_dates_for_alignment = None # Store dates for aligning predictions
    try:
        # Menggunakan Pandas untuk membaca data (Membaca)
        data_path = config['data']['raw_path']
        if not os.path.exists(data_path):
             logger.error(f"File data tidak ditemukan di {os.path.abspath(data_path)}. Pastikan path benar dan file ada.")
             return # Hentikan pipeline jika file data tidak ada

        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Format data tidak didukung. Gunakan .csv atau .json.")

        logger.info(f"Data mentah dimuat dari {data_path}. Jumlah baris: {len(df)}")

        # Pastikan data terurut berdasarkan Date jika ada (Opsional tapi direkomendasikan)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']) # Konversi ke datetime
            df.sort_values(by='Date', inplace=True)
            logger.info("Data diurutkan berdasarkan kolom 'Date'.")
            # Simpan kolom 'Date' asli untuk alignment prediksi nanti
            original_dates_for_alignment = df['Date'].copy()
        else:
            logger.warning("Kolom 'Date' tidak ditemukan. Prediksi tidak akan bisa di-align dengan tanggal asli.")


        # Pembersihan data awal
        initial_rows = len(df)
        # Hapus baris dengan NaN di kolom harga atau kolom fitur numerik yang *dipastikan* ada
        cols_to_check_nan = config['data']['feature_cols_numeric'] # Pastikan ini termasuk OHLCV
        if 'Date' in df.columns and 'include_date_in_nan_check' in config['data'] and config['data']['include_date_in_nan_check']:
             cols_to_check_nan.append('Date') # Tambahkan Date jika konfigurasi memintanya
             logger.info("Menyertakan kolom 'Date' dalam pengecekan NaN.")

        df.dropna(subset=cols_to_check_nan, inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di kolom: {cols_to_check_nan}") # Log kolom yg dicek


        # Pastikan kolom OHLC ada untuk menghitung Pivot Points
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
             # Feature Engineering (Menggunakan Aritmatika)
             # Contoh: Menghitung Pivot Points
             logger.info("Menghitung Pivot Points...")
             # Menggunakan Aritmatika Pandas Series
             df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
             df['R1'] = 2 * df['Pivot'] - df['Low']
             df['S1'] = 2 * df['Pivot'] - df['High']
             df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
             df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
             df['R3'] = df['Pivot'] + 2 * (df['High'] - df['Low'])
             df['S3'] = df['Pivot'] - 2 * (df['High'] - df['Low'])
             # Tambahkan R4/R5, S4/S5 jika perlu
             logger.info("Perhitungan Pivot Points selesai.")
        else:
             logger.warning("Kolom OHLC tidak lengkap untuk menghitung Pivot Points.")


        # Menyiapkan Target (HLC Selanjutnya)
        logger.info(f"Menyiapkan target HLC selanjutnya dengan shift -{config['parameter_windowing']['window_size']}...")
        # Menggunakan Pandas shift. Shift negatif menarik data masa depan ke baris saat ini.
        # Jendela input size=N di baris T akan memprediksi target di baris T+N.
        # Jadi, target di baris T haruslah HLC di baris T+N.
        # shift(-N) menggeser nilai dari N baris ke bawah ke baris saat ini.
        # Target columns are ['High_Next', 'Low_Next', 'Close_Next']
        target_cols = ['High_Next', 'Low_Next', 'Close_Next']
        df['High_Next'] = df['High'].shift(-config['parameter_windowing']['window_size'])
        df['Low_Next'] = df['Low'].shift(-config['parameter_windowing']['window_size'])
        df['Close_Next'] = df['Close'].shift(-config['parameter_windowing']['window_size'])

        # Menghapus baris dengan NaN di target setelah shift
        # Jumlah baris yang dihapus akan sama dengan window_size
        initial_rows = len(df)
        df.dropna(subset=target_cols, inplace=True)
        if len(df) < initial_rows:
             logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di target setelah shift (-{config['parameter_windowing']['window_size']}).")


        # Membuat Fitur Teks Deskriptif (Menulis Teks ke Kolom)
        feature_cols_text = []
        if config['use_text_input']:
            logger.info("Membuat fitur teks deskriptif...")
            # Menggunakan Pandas apply dan fungsi kustom generate_analysis_text
            # Pastikan kolom yang dibutuhkan generate_analysis_text ada setelah dropna
            # Adjust check based on columns used in generate_analysis_text
            cols_needed_for_text = ['High', 'Low', 'Close', 'Pivot', 'R1', 'S1'] # Example columns
            if all(col in df.columns for col in cols_needed_for_text):
                df['Analysis_Text'] = df.apply(generate_analysis_text, axis=1)
                feature_cols_text = ['Analysis_Text']
                if not df.empty:
                     logger.info(f"Contoh teks analisis: {df['Analysis_Text'].iloc[0]}")
                else:
                     logger.warning("DataFrame kosong setelah membuat teks analisis.")
            else:
                 logger.warning(f"Kolom yang dibutuhkan untuk generate_analysis_text tidak lengkap setelah preprocessing: {cols_needed_for_text}. Fitur teks tidak akan dibuat.")


        # Identifikasi Fitur Input dan Target
        # Pastikan kolom indikator yang dihitung juga masuk fitur numerik
        indicator_cols = [col for col in df.columns if col.startswith(('Piv', 'R', 'S'))] # Match 'Pivot', 'R*', 'S*'
        # Ambil feature_cols_numeric dari config, lalu tambahkan indicator_cols jika belum ada
        feature_cols_numeric = list(config['data']['feature_cols_numeric']) # Mulai dengan kolom dari config
        for col in indicator_cols:
            if col not in feature_cols_numeric and col in df.columns: # Only add if calculated column exists in df
                feature_cols_numeric.append(col) # Tambahkan kolom indikator yang dihitung

        # Filter DataFrame hanya untuk kolom fitur dan target yang relevan
        # Exclude 'Date' explicitly from this processing DataFrame if it exists
        all_feature_target_cols = feature_cols_numeric + feature_cols_text + target_cols
        if 'Date' in all_feature_target_cols:
             all_feature_target_cols.remove('Date') # Ensure Date is not in features/targets here

        df_processed = df[all_feature_target_cols].copy() # Buat salinan untuk menghindari SettingWithCopyWarning


        # Konversi ke NumPy Arrays
        # Menggunakan NumPy untuk konversi
        if not df_processed.empty:
            # Konversi hanya kolom fitur dan target yang relevan
            data_numeric = df_processed[feature_cols_numeric].values.astype(np.float32)
            data_text = df_processed[feature_cols_text].values.flatten().astype(str) if feature_cols_text else np.array([], dtype=str)
            data_target = df_processed[target_cols].values.astype(np.float32)

            logger.info(f"Final data numerik shape: {data_numeric.shape}")
            logger.info(f"Final data teks shape: {data_text.shape}")
            logger.info(f"Final data target shape: {data_target.shape}")

            # Store the index splits for later date alignment
            total_samples_processed = len(df_processed)
            train_size = int(np.floor(total_samples_processed * config['data']['train_split']))
            val_size = int(np.floor(total_samples_processed * config['data']['val_split']))
            test_size = total_samples_processed - train_size - val_size # Ukuran test adalah sisanya

            # Store test set dates BEFORE tf.data creation
            # These dates correspond to the data points AFTER target shift and dropna
            # We need dates corresponding to the *predictions*, which are N steps ahead
            # We will get these dates from the original_dates_for_alignment series later
            test_split_start_idx_processed_df = train_size + val_size # Index in df_processed where test set begins


        else:
            logger.error("DataFrame kosong setelah preprocessing. Tidak dapat melanjutkan.")
            return # Hentikan pipeline jika data kosong setelah preprocessing


        # Membagi Data (menggunakan slicing NumPy)
        # train_size, val_size, test_size defined above based on df_processed length
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
             logger.error(f"Ukuran set data tidak memadai. Train: {train_size}, Val: {val_size}, Test: {test_size}. Sesuaikan split atau gunakan data lebih banyak.")
             return # Hentikan pipeline jika ukuran data tidak memadai


        input_train_numeric = data_numeric[:train_size]
        input_val_numeric = data_numeric[train_size:train_size + val_size]
        input_test_numeric = data_numeric[train_size + val_size:]

        input_train_text = data_text[:train_size] if feature_cols_text else np.array([], dtype=str)
        input_val_text = data_text[train_size:train_size + val_size] if feature_cols_text else np.array([], dtype=str)
        input_test_text = data_text[train_size + val_size:] if feature_cols_text else np.array([], dtype=str)

        target_train = data_target[:train_size]
        target_val = data_target[train_size:train_size + val_size]
        target_test = data_target[train_size + val_size:] # Keep original scale for evaluation

        logger.info(f"Train set size: {len(input_train_numeric)}")
        logger.info(f"Validation set size: {len(input_val_numeric)}")
        logger.info(f"Test set size: {len(input_test_numeric)}")

    except Exception as e:
        logger.error(f"Error selama Langkah 1: {e}")
        # Log the specific error if possible
        # logger.error(f"Traceback: {traceback.format_exc()}") # Requires import traceback
        return # Hentikan pipeline jika ada error fatal

    # (Langkah 2, 3, 4, 5 remain largely the same, with minor adjustments for text input shape and vocabulary)
    # --- Langkah 2: Scaling Data & Pembuatan Pipeline tf.data ---
    logger.info("Langkah 2: Scaling data dan membuat pipeline tf.data...")
    scaler_input = None # Inisialisasi scaler
    scaler_target = None
    vocabulary_table = None
    vocabulary_size = None
    try:
        # Scaling Data Numerik (Menggunakan Aritmatika di dalam scaler)
        # Only scale if there is numeric data
        if input_train_numeric.size > 0:
            scaler_input = MinMaxScaler()
            scaler_target = MinMaxScaler()

            # Latih scaler hanya pada data pelatihan
            scaled_input_train = scaler_input.fit_transform(input_train_numeric)
            scaled_target_train = scaler_target.fit_transform(target_train)

            # Terapkan scaler pada data validasi dan pengujian
            scaled_input_val = scaler_input.transform(input_val_numeric)
            scaled_input_test = scaler_input.transform(input_test_numeric)

            scaled_target_val = scaler_target.transform(target_val)
            # target_test tidak perlu diskalakan di sini untuk evaluasi/inverse transform nanti,
            # hanya input_test_numeric yang diskalakan untuk dimasukkan ke model.
            logger.info("Data numerik berhasil diskalakan.")
        else:
             logger.warning("Tidak ada data numerik untuk scaling.")
             # Handle cases where there's no numeric data but maybe only text input? (Unlikely for this model)
             scaled_input_train = input_train_numeric
             scaled_input_val = input_val_numeric
             scaled_input_test = input_test_numeric
             scaled_target_train = target_train
             scaled_target_val = target_val
             # Scalers will remain None


        # Membuat Kosakata untuk Teks (jika digunakan)
        if config['use_text_input'] and feature_cols_text and input_train_text.size > 0:
            logger.info("Membuat kosakata dari data pelatihan teks...")
            # Mengumpulkan semua token unik dari data pelatihan teks
            input_train_text_str = tf.constant(input_train_text.tolist(), dtype=tf.string)
            all_train_tokens_ragged = tf.strings.split(input_train_text_str)
            all_train_tokens_np = all_train_tokens_ragged.values.numpy()
            unique_tokens_np = np.unique(all_train_tokens_np[all_train_tokens_np != b''])

            keys = tf.constant(unique_tokens_np, dtype=tf.string)
            values = tf.range(1, tf.size(keys) + 1, dtype=tf.int64)
            num_oov_buckets = 1 # Reserve one bucket for OOV
            init = tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int64)
            vocabulary_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=num_oov_buckets)
            vocabulary_size = vocabulary_table.size().numpy() # Size includes unique keys + num_oov_buckets
            logger.info(f"Ukuran kosakata teks (termasuk OOV): {vocabulary_size}")
            if len(unique_tokens_np) > 0:
                 logger.info(f"Contoh token unik (mapping value): {[t.decode('utf-8') for t in unique_tokens_np[:5]]} -> {vocabulary_table.lookup(tf.constant(unique_tokens_np[:5])).numpy()}")
        elif config['use_text_input'] and feature_cols_text:
             logger.warning("Fitur teks diaktifkan tetapi data pelatihan teks kosong. Kosakata tidak dibuat.")
        elif config['use_text_input'] and not feature_cols_text:
             logger.warning("Fitur teks diaktifkan di config, tetapi kolom teks ('Analysis_Text') tidak ditemukan atau tidak dibuat.")
        else:
             logger.info("Fitur teks tidak digunakan.")


        # Membuat tf.data.Dataset dari data yang sudah diproses
        # Menggunakan API TensorFlow Data: from_tensor_slices
        # Need to handle case where text_data is empty NumPy array
        if feature_cols_text: # If text input is configured AND text column was created
            dataset_train_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_train, input_train_text), scaled_target_train))
            dataset_val_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_val, input_val_text), scaled_target_val))
            dataset_test_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_test, input_test_text), target_test)) # Target test not scaled
        else:
            # If no text input, structure is just (numeric_features, target)
            dataset_train_raw = tf.data.Dataset.from_tensor_slices((scaled_input_train, scaled_target_train))
            dataset_val_raw = tf.data.Dataset.from_tensor_slices((scaled_input_val, scaled_target_val))
            dataset_test_raw = tf.data.Dataset.from_tensor_slices((scaled_input_test, target_test))


        # Fungsi untuk membuat window dan memproses elemen dataset
        @tf.function
        def process_window_elements(input_elements, target_window, use_text, max_seq_len, vocab_table):
             """Memproses window data (termasuk teks) di inside pipeline map."""
             # input_elements will be (numeric_window, text_window) if use_text is True, else just numeric_window
             if use_text:
                 numeric_window, text_window = input_elements
             else:
                 numeric_window = input_elements
                 text_window = None # Not used

             # target_window shape: (window_size, num_target_features) - we only need the LAST target!
             # Use Aritmatika Indexing
             final_target = target_window[-1]

             processed_text_window = None # Default

             if use_text and text_window is not None and vocab_table is not None:
                 # Process each text element in the window
                 processed_text_window = tf.map_fn(
                     lambda text_elem: process_text_for_model(text_elem, max_seq_len, vocab_table),
                     text_window,
                     fn_output_signature=tf.int64
                 )
                 # processed_text_window shape should now be (window_size, max_text_sequence_length)

             # Return tuple that matches model input structure and the *final* target
             if use_text and processed_text_window is not None:
                 return (numeric_window, processed_text_window), final_target
             else:
                 return numeric_window, final_target


        # Functions to create windowed datasets
        def create_window_dataset(dataset_raw, window_size, window_shift, drop_remainder):
             dataset_windowed = dataset_raw.window(size=window_size, shift=window_shift, drop_remainder=drop_remainder)
             dataset_flattened = dataset_windowed.flat_map(lambda window: window.batch(window_size))
             return dataset_flattened

        # Get processing parameters for the map function
        map_params = {
            'use_text': config['use_text_input'] and feature_cols_text, # Only use text if configured AND column exists
            'max_seq_len': config['data'].get('max_text_sequence_length', 20), # Use .get with default for safety
            'vocab_table': vocabulary_table
        }


        # Membuat Pipeline tf.data (Windowing, Batching, Caching, Prefetching)
        # Pipeline Pelatihan
        dataset_train_windowed = create_window_dataset(dataset_train_raw, config['parameter_windowing']['window_size'], config['parameter_windowing']['window_shift'], True)
        # Pass parameters to the map function
        dataset_train = dataset_train_windowed.map(
            lambda input_elements, target_window: process_window_elements(input_elements, target_window, **map_params),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset_train = dataset_train.shuffle(config['data']['shuffle_buffer_size'])
        dataset_train = dataset_train.batch(config['training']['batch_size']).cache().prefetch(tf.data.AUTOTUNE)
        logger.info(f"Pipeline train tf.data: {dataset_train.element_spec}") # Log element spec

        # Pipeline Validasi (tanpa shuffle)
        dataset_val_windowed = create_window_dataset(dataset_val_raw, config['parameter_windowing']['window_size'], config['parameter_windowing']['window_shift'], True)
        dataset_val = dataset_val_windowed.map(
             lambda input_elements, target_window: process_window_elements(input_elements, target_window, **map_params),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset_val = dataset_val.batch(config['training']['batch_size']).cache().prefetch(tf.data.AUTOTUNE)
        logger.info(f"Pipeline val tf.data: {dataset_val.element_spec}") # Log element spec


        # Pipeline Pengujian (tanpa shuffle, tanpa cache jika data test besar)
        dataset_test_windowed = create_window_dataset(dataset_test_raw, config['parameter_windowing']['window_size'], config['parameter_windowing']['window_shift'], True)
        dataset_test = dataset_test_windowed.map(
            lambda input_elements, target_window: process_window_elements(input_elements, target_window, **map_params),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset_test = dataset_test.batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
        logger.info(f"Pipeline test tf.data: {dataset_test.element_spec}") # Log element spec


        logger.info("Pipeline tf.data selesai dibuat.")

    except Exception as e:
        logger.error(f"Error selama Langkah 2: {e}")
        # logger.error(f"Traceback: {traceback.format_exc()}") # Requires import traceback
        return # Hentikan pipeline jika ada error fatal


    # --- Langkah 3: Definisi & Kompilasi Model ---
    # (Model definition and compilation remains the same, adjusted for text input shape)
    logger.info("Langkah 3: Mendefinisikan atau memuat model...")
    model = None
    # Define model architecture parameters
    numeric_input_shape = (config['parameter_windowing']['window_size'], len(feature_cols_numeric))
    # Text input shape is (window_size, max_text_sequence_length) after processing, if text used
    text_input_shape = (config['parameter_windowing']['window_size'], config['data'].get('max_text_sequence_length', 20)) if config['use_text_input'] and feature_cols_text else None
    num_target_features = len(target_cols)

    try:
        if config['mode'] == 'initial_train':
            logger.info("Mode: initial_train. Mendefinisikan model baru.")
            model = build_quantai_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size)

            logger.info("Mengkompilasi model.")
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
            loss_fn = tf.keras.losses.MeanAbsoluteError()
            metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

            logger.info("Model baru berhasil didefinisikan dan dikompilasi.")
            model.summary(print_fn=logger.info)

        elif config['mode'] in ['incremental_learn', 'predict_only']:
            load_path = config['model'].get('load_path') # Use .get for safety
            if not load_path:
                logger.error(f"Mode '{config['mode']}' dipilih, tetapi model.load_path tidak ada di konfigurasi.")
                return # Hentikan pipeline

            logger.info(f"Mode: {config['mode']}. Memuat model dari {load_path}")
            if not os.path.exists(load_path):
                 logger.error(f"File model tidak ditemukan di {os.path.abspath(load_path)}. Pastikan path benar dan model telah disimpan sebelumnya.")
                 return # Hentikan pipeline jika model tidak ditemukan untuk dimuat

            try:
                model = tf.keras.models.load_model(load_path) # Assumes .h5 or SavedModel compatible with load_model
                logger.info(f"Model berhasil dimuat dari {load_path}.")
            except Exception as e:
                logger.error(f"Gagal memuat model dari {load_path}: {e}")
                model = None # Set model ke None jika gagal memuat

            if model is not None:
                if config['mode'] == 'incremental_learn':
                    logger.info("Mengkompilasi ulang model untuk incremental_learn.")
                    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
                    loss_fn = tf.keras.losses.MeanAbsoluteError()
                    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
                    logger.info("Model berhasil dikompilasi ulang.")

        else:
            logger.error(f"Mode operasional tidak valid: {config['mode']}")
            return # Hentikan pipeline jika mode tidak valid

    except Exception as e:
        logger.error(f"Error selama Langkah 3: {e}")
        # logger.error(f"Traceback: {traceback.format_exc()}") # Requires import traceback
        return # Hentikan pipeline jika ada error fatal

    if model is None:
         logger.error("Model tidak tersedia setelah Langkah 3. Menghentikan pipeline.")
         return

    # --- Langkah 4: Pelatihan Model (Belajar Mandiri/Hybrid/Otonom) ---
    # (Training logic remains the same)
    if config['mode'] in ['initial_train', 'incremental_learn']:
        logger.info(f"Langkah 4: Memulai pelatihan model dalam mode {config['mode']}...")
        try:
            output_dir = config['output']['base_dir']
            model_save_file = config['output']['model_save_file']
            model_save_path_full = os.path.join(output_dir, model_save_file) # Use model_save_file
            scaler_save_dir_full = os.path.join(output_dir, config['output']['scaler_subdir'])
            tensorboard_log_dir_full = os.path.join(output_dir, config['output']['tensorboard_log_dir'])

            tf.io.gfile.makedirs(os.path.dirname(model_save_path_full))
            tf.io.gfile.makedirs(scaler_save_dir_full)
            tf.io.gfile.makedirs(tensorboard_log_dir_full)


            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'], restore_best_weights=True),
                # Save model in .h5 format
                tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path_full, monitor='val_loss', save_best_only=True, save_format='h5'), # <-- save_format='h5'
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['training']['lr_reduce_factor'], patience=config['training']['lr_reduce_patience'], min_lr=config['training']['min_lr']),
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
            logger.info(f"Memuat model terbaik dari {model_save_path_full}")
            try:
                model = tf.keras.models.load_model(model_save_path_full)
                logger.info("Model terbaik berhasil dimuat setelah pelatihan.")
            except Exception as e:
                 logger.warning(f"Gagal memuat model terbaik setelah pelatihan dari {model_save_path_full}: {e}. Menggunakan model akhir dari fit().")


            # --- Fondasi Belajar Mandiri/Hybrid/Otonom (Loop Kustom Opsional) ---
            # (Pseudocode placeholders remain)

        except Exception as e:
            logger.error(f"Error selama Langkah 4: {e}")
            # logger.error(f"Traceback: {traceback.format_exc()}") # Requires import traceback
            # Lanjutkan ke langkah berikutnya meskipun ada error pelatihan


    # --- Langkah 5: Evaluation Model Akhir ---
    # (Evaluation logic remains the same)
    eval_results = None
    if config['mode'] in ['initial_train', 'incremental_learn'] and model is not None:
        logger.info("Langkah 5: Mengevaluasi model akhir...")
        try:
            if hasattr(model, 'evaluate') and 'dataset_test' in locals():
                logger.info(f"Dataset test size untuk evaluasi: {tf.data.Dataset.cardinality(dataset_test).numpy()} batches")
                eval_results = model.evaluate(dataset_test)
                if hasattr(model, 'metrics_names'):
                     logger.info(f"Hasil Evaluasi Akhir: {dict(zip(model.metrics_names, eval_results))}")
                else:
                     logger.info(f"Hasil Evaluasi Akhir (metrik tidak tersedia): {eval_results}")

            elif 'dataset_test' not in locals():
                 logger.warning("Dataset test tidak tersedia. Tidak dapat melakukan evaluasi.")
            else:
                 logger.error("Model tidak memiliki metode evaluate. Tidak dapat melakukan evaluasi.")

        except Exception as e:
            logger.error(f"Error selama Langkah 5: {e}")
            # logger.error(f"Traceback: {traceback.format_exc()}") # Requires import traceback
            # Lanjutkan ke langkah berikutnya meskipun ada error evaluasi


    # --- Langkah 6: Penyimpanan Aset & Hasil (Menulis) ---
    logger.info("Langkah 6: Menyimpan aset dan hasil...")
    try:
        output_dir = config['output']['base_dir']
        # model_save_file path handled by ModelCheckpoint in Step 4
        scaler_save_dir = config['output']['scaler_subdir']
        eval_results_file = config['output']['eval_results_file']
        predictions_file = config['output']['predictions_file']
        # tensorboard_log_dir handled in Step 4

        # Create necessary directories
        tf.io.gfile.makedirs(os.path.join(output_dir, scaler_save_dir))
        tf.io.gfile.makedirs(os.path.dirname(os.path.join(output_dir, eval_results_file)))
        tf.io.gfile.makedirs(os.path.dirname(os.path.join(output_dir, predictions_file)))


        if config['mode'] in ['initial_train', 'incremental_learn']:
            logger.info(f"Model terbaik dalam format .h5 sudah disimpan oleh ModelCheckpoint di Langkah 4.")

            # Menyimpan Scaler
            logger.info(f"Menyimpan scaler ke {os.path.join(output_dir, scaler_save_dir)}...")
            if 'scaler_input' in locals() and scaler_input is not None and 'scaler_target' in locals() and scaler_target is not None:
                 joblib.dump(scaler_input, os.path.join(output_dir, scaler_save_dir, 'scaler_input.pkl'))
                 joblib.dump(scaler_target, os.path.join(output_dir, scaler_save_dir, 'scaler_target.pkl'))
                 logger.info("Scaler berhasil disimpan.")
            else:
                 logger.warning("Scaler tidak tersedia. Tidak dapat menyimpan scaler.")

            # Menyimpan Kosakata Teks
            if config['use_text_input'] and feature_cols_text and 'vocabulary_table' in locals() and vocabulary_table is not None:
                 vocab_file_path = os.path.join(output_dir, config['output'].get('vocabulary_file', os.path.join(scaler_save_dir, 'vocabulary.txt'))) # Use default path if not in config
                 tf.io.gfile.makedirs(os.path.dirname(vocab_file_path))
                 try:
                    if 'unique_tokens_np' in locals() and unique_tokens_np is not None:
                        with open(vocab_file_path, 'w', encoding='utf-8') as f:
                            for token in unique_tokens_np:
                                f.write(token.decode('utf-8') + '\n')
                        logger.info(f"Kosakata teks berhasil disimpan ke {vocab_file_path}.")
                    else:
                         logger.warning("Daftar token unik untuk kosakata tidak tersedia. Tidak dapat menyimpan kosakata.")
                 except Exception as e:
                     logger.warning(f"Gagal menyimpan kosakata teks: {e}")


            # Menyimpan Hasil Evaluasi
            if eval_results is not None and model is not None and hasattr(model, 'metrics_names'):
                logger.info(f"Menyimpan hasil evaluasi ke {os.path.join(output_dir, eval_results_file)}...")
                eval_dict = dict(zip(model.metrics_names, eval_results))
                with open(os.path.join(output_dir, eval_results_file), 'w') as f:
                    json.dump(eval_dict, f, indent=4)
                logger.info("Hasil evaluasi berhasil disimpan.")
            elif eval_results is not None:
                 logger.warning("Nama metrik model tidak tersedia. Tidak dapat menyimpan hasil evaluasi dalam format dictionary.")


        # Menyimpan Prediksi (Output Hasil, termasuk Date)
        # Hanya simpan prediksi jika save_predictions True DAN model berhasil dimuat/dilatih
        if config['output'].get('save_predictions', False) and model is not None:
             # Use dataset_test for prediction
             if 'dataset_test' in locals() and dataset_test is not None:
                logger.info(f"Membuat dan menyimpan prediksi ke {os.path.join(output_dir, predictions_file)}...")
                # Menggunakan API Keras Model: predict
                predictions_scaled = model.predict(dataset_test)

                # Pastikan scaler_target tersedia untuk inverse transform
                if 'scaler_target' in locals() and scaler_target is not None:
                    # Menggunakan Aritmatika di dalam scaler inverse_transform
                    predictions_original_scale = scaler_target.inverse_transform(predictions_scaled)

                    # Membuat DataFrame hasil prediksi
                    df_predictions = pd.DataFrame(predictions_original_scale, columns=[f'{col}_Pred' for col in target_cols])

                    # --- Menambahkan Kolom Date ke Hasil Prediksi ---
                    # Align predictions with original dates.
                    # Predictions correspond to the target N steps ahead of the window end.
                    # The test split in df_processed starts at index train_size + val_size.
                    # The first window uses data up to index (train_size + val_size) + window_size - 1 in original df.
                    # The prediction for this window is for the target at index (train_size + val_size) + window_size in original df.
                    # We need the dates from original_dates_for_alignment starting from this index.

                    if original_dates_for_alignment is not None:
                         # Calculate the starting index in the original dates series for the predictions
                         # The targets in df_processed align with index train_size + val_size onwards.
                         # These targets were shifted by -window_size.
                         # So, the prediction for the data at index i in df_processed is for the date originally at index i + window_size.
                         # The first index in df_processed test set is test_split_start_idx_processed_df (which is train_size + val_size).
                         # The corresponding date in original_dates_for_alignment is at index test_split_start_idx_processed_df + window_size.

                         date_start_index_for_predictions = test_split_start_idx_processed_df + config['parameter_windowing']['window_size']
                         # Ensure we don't go out of bounds of the original dates series
                         if date_start_index_for_predictions < len(original_dates_for_alignment):
                            # Select the dates corresponding to the predictions
                            # The number of predictions should match the number of windows in the test set.
                            num_predictions = len(df_predictions)
                            predicted_dates = original_dates_for_alignment.iloc[
                                date_start_index_for_predictions : date_start_index_for_predictions + num_predictions * config['parameter_windowing']['window_shift'] # Account for window shift
                            ].reset_index(drop=True) # Reset index to align with predictions df

                            # Check if the number of dates matches the number of predictions
                            if len(predicted_dates) == num_predictions:
                                 df_predictions.insert(0, 'Date', predicted_dates) # Insert Date as the first column
                                 logger.info("Kolom 'Date' berhasil ditambahkan ke hasil prediksi.")
                            else:
                                 logger.warning(f"Jumlah prediksi ({num_predictions}) tidak sesuai dengan jumlah tanggal yang diambil ({len(predicted_dates)}). Tidak dapat menambahkan kolom Date dengan benar.")
                                 logger.warning("Ini bisa terjadi jika window_shift tidak 1 atau ada kompleksitas indexing lainnya.")
                                 # Proceed without Date column in predictions
                                 pass # Keep df_predictions as is

                         else:
                              logger.warning("Indeks awal tanggal untuk prediksi di luar rentang tanggal asli. Tidak dapat menambahkan kolom Date.")
                              # Proceed without Date column in predictions
                              pass

                    else:
                         logger.warning("Kolom 'Date' asli tidak tersedia untuk alignment prediksi. Tidak dapat menambahkan kolom Date.")
                         # Proceed without Date column in predictions
                         pass


                    # Menggunakan Pandas to_csv (Menulis)
                    # Save to the predictions_file path constructed relative to output_dir
                    df_predictions.to_csv(os.path.join(output_dir, predictions_file), index=False) # Set index=True jika ingin menyimpan index Date
                    logger.info("Prediksi berhasil disimpan.")
                elif model is None:
                     logger.warning("Model tidak tersedia. Tidak dapat membuat prediksi.")
                elif ('scaler_target' in locals() and scaler_target is None) or ('scaler_target' not in locals()):
                     logger.warning("Scaler target tidak tersedia. Tidak dapat melakukan inverse transform atau menyimpan prediksi.")

             elif 'dataset_test' not in locals() or dataset_test is None:
                  logger.warning("Dataset test tidak tersedia. Tidak dapat membuat prediksi.")
             else:
                  logger.warning("Prediksi tidak disimpan karena save_predictions diatur ke False di konfigurasi.")


        logger.info("Langkah 6 selesai.")

    except Exception as e:
        logger.error(f"Error selama Langkah 6: {e}")
        # logger.error(f"Traceback: {traceback.format_exc()}") # Requires import traceback


    logger.info("Pipeline QuantAI selesai.")

# --- Eksekusi Pipeline ---

if __name__ == "__main__":
    import traceback # Import traceback module here for detailed logging

    parser = argparse.ArgumentParser(description='Run QuantAI ML Pipeline.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config_path = args.config

    config = None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Konfigurasi berhasil dimuat dari {config_path}")

    except FileNotFoundError:
        logger.error(f"File konfigurasi tidak ditemukan di {config_path}. Pastikan path benar di workflow atau command-line.")
        exit(1)

    except yaml.YAMLError as e:
        logger.error(f"Error mem-parsing file konfigurasi YAML dari {config_path}: {e}")
        exit(1)

    if config:
        # --- Validasi Konfigurasi Esensial ---
        essential_keys = ['data', 'model', 'training', 'output', 'parameter_windowing', 'seed'] # Added seed
        missing_keys = [key for key in essential_keys if key not in config or config[key] is None]

        if missing_keys:
            logger.error(f"File konfigurasi ({config_path}) tidak lengkap atau rusak. Kunci utama yang hilang atau kosong: {missing_keys}")
            logger.error("Pastikan file konfigurasi YAML memiliki semua bagian utama (data, model, training, output, parameter_windowing, seed) dan tidak kosong.")
            exit(1)

        # --- Menambahkan Path Default (jika belum ada) ---
        if isinstance(config.get('output'), dict): # Check if 'output' exists and is a dict
             if config.get('use_text_input', False) and 'vocabulary_file' not in config['output']:
                  scaler_subdir = config['output'].get('scaler_subdir', 'scalers')
                  base_dir = config['output'].get('base_dir', '')
                  config['output']['vocabulary_file'] = os.path.join(base_dir, scaler_subdir, 'vocabulary.txt')
                  logger.info(f"Menambahkan path default vocabulary_file: {config['output']['vocabulary_file']}")

             if 'model_save_file' not in config['output']:
                  model_subdir = config['output'].get('model_subdir', 'saved_model')
                  base_dir = config['output'].get('base_dir', '')
                  config['output']['model_save_file'] = os.path.join(base_dir, model_subdir, 'best_model.h5')
                  logger.info(f"Menambahkan path default model_save_file: {config['output']['model_save_file']}")
        # No else needed, validation caught it earlier

        # --- Validasi Mode dan load_path ---
        if config['mode'] in ['incremental_learn', 'predict_only']:
            if not isinstance(config.get('model'), dict) or 'load_path' not in config['model'] or not config['model']['load_path']:
                 logger.error(f"Mode '{config['mode']}' dipilih, tetapi kunci 'model' tidak lengkap atau 'model.load_path' tidak ada/kosong di konfigurasi.")
                 exit(1)
            # Optional: Warning if load_path doesn't end with .h5 if saving as .h5
            elif isinstance(config.get('output'), dict) and 'model_save_file' in config['output'] and config['output']['model_save_file'].endswith('.h5') and not config['model']['load_path'].endswith('.h5'):
                  logger.warning(f"Mode '{config['mode']}' dipilih dan model disimpan sebagai .h5, tetapi model.load_path ({config['model']['load_path']}) tidak berakhir dengan .h5. Pastikan path memuat file .h5 yang benar.")


        # --- Jalankan Pipeline Utama ---
        # Base directory must exist in output config
        if isinstance(config.get('output'), dict) and 'base_dir' in config['output']:
             run_pipeline(config)
        else:
             # This should have been caught by essential_keys, but as fallback
             logger.error("Konfigurasi output atau base_dir tidak ditemukan atau salah format setelah validasi.")
             exit(1)

    else:
        logger.error("Tidak ada konfigurasi yang tersedia setelah mencoba memuat file.")
        exit(1)

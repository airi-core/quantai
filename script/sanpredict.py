import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K # K didefinisikan di sini
import numpy as np
import os
import shutil # Jika Anda ingin membersihkan folder model nanti

NUM_CLASSES = 9
TOTAL_SUM = 45

# --- Fungsi Helper untuk Lambda Layers ---
def sum_inputs_keras(x_input): # Ubah nama parameter agar tidak bentrok dengan 'x' di lambda
    return K.sum(x_input, axis=-1, keepdims=True)

def calculate_missing_digit_keras(sum_result): # Parameter diubah agar jelas
    return TOTAL_SUM - sum_result

def calculate_index_keras(missing_digit_result): # Parameter diubah agar jelas
    # Pastikan input ke K.round adalah float, jika tidak bisa error di beberapa versi TF
    float_input = K.cast(missing_digit_result, K.floatx())
    rounded_val = K.round(float_input)
    index_val = rounded_val - 1
    clipped_val = K.clip(index_val, 0, NUM_CLASSES - 1)
    return K.cast(clipped_val, 'int32')

def to_one_hot_keras(index_input): # Parameter diubah agar jelas
    # Pastikan input ke K.one_hot adalah int32 dan diratakan
    flattened_index = K.flatten(index_input)
    int32_index = K.cast(flattened_index, 'int32')
    return K.one_hot(int32_index, NUM_CLASSES)
# --- Akhir Fungsi Helper ---

def build_model_jumlah_sederhana():
    inputs = Input(shape=(8,), name="delapan_angka")

    # Menggunakan fungsi helper yang sudah didefinisikan
    sum_layer = Lambda(sum_inputs_keras, name="jumlah_input", output_shape=(1,))(inputs)
    missing_digit_layer = Lambda(calculate_missing_digit_keras, name="angka_hilang", output_shape=(1,))(sum_layer)
    index_layer = Lambda(calculate_index_keras, name="indeks", output_shape=(1,))(missing_digit_layer)
    outputs = Lambda(to_one_hot_keras, name="hasil_prediksi", output_shape=(NUM_CLASSES,))(index_layer)

    model = Model(inputs=inputs, outputs=outputs, name="Model_Rumus_Eksplisit")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model_pembelajaran():
    model = keras.Sequential(name="Model_Pembelajaran_Sederhana")
    model.add(Input(shape=(8,), name="input_layer"))
    model.add(Dense(units=1, name="dense_sum_like")) # Aktivasi default adalah linear
    model.add(Dense(units=16, activation='relu', name="dense_hidden"))
    model.add(Dense(units=NUM_CLASSES, activation='softmax', name="output_probs"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_training_data(n_samples=10000):
    X_data = []
    y_data = []
    all_possible_numbers = list(range(1, 10))

    for _ in range(n_samples):
        missing_number = np.random.randint(1, 10)
        current_numbers = list(all_possible_numbers) # Buat salinan
        current_numbers.remove(missing_number)
        np.random.shuffle(current_numbers)
        X_data.append(current_numbers)

        one_hot_label = np.zeros(NUM_CLASSES)
        one_hot_label[missing_number - 1] = 1
        y_data.append(one_hot_label)

    return np.array(X_data), np.array(y_data)

def save_and_test_models():
    print("Memulai pembuatan, pelatihan, dan penyimpanan model angka hilang dalam format H5...")

    models_dir = 'models_h5'
    model_rumus_path = os.path.join(models_dir, 'model_rumus_eksplisit.h5')
    model_belajar_path = os.path.join(models_dir, 'model_pembelajaran.h5')

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Folder '{models_dir}' dibuat.")
    # else: # Opsi: bersihkan folder jika sudah ada untuk pengujian ulang yang bersih
    #     print(f"Membersihkan folder '{models_dir}'...")
    #     shutil.rmtree(models_dir)
    #     os.makedirs(models_dir)


    print("\nMembuat Model Rumus Eksplisit...")
    model_rumus = build_model_jumlah_sederhana()
    model_rumus.summary()
    try:
        model_rumus.save(model_rumus_path)
        print(f"Model rumus eksplisit disimpan sebagai file '{model_rumus_path}'")
    except Exception as e:
        print(f"\nERROR saat menyimpan model rumus eksplisit ke H5: {e}")

    print("\nMembuat Model Pembelajaran...")
    model_belajar = build_model_pembelajaran()
    model_belajar.summary()

    print("\nMenghasilkan data pelatihan...")
    X_train, y_train = generate_training_data(n_samples=50000) # Meningkatkan jumlah sampel

    print(f"\nMelatih Model Pembelajaran dengan {X_train.shape[0]} sampel...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Menggunakan sebagian kecil data untuk validasi agar lebih cepat, namun idealnya data validasi terpisah
    history = model_belajar.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    try:
        model_belajar.save(model_belajar_path)
        print(f"Model pembelajaran disimpan sebagai file '{model_belajar_path}'")
    except Exception as e:
        print(f"\nERROR saat menyimpan model pembelajaran ke H5: {e}")

    print("\nProses pembuatan, pelatihan, dan penyimpanan selesai (jika tidak ada error penyimpanan).")

    print("\n--- Memuat kembali model yang tersimpan dari H5 ---")
    loaded_model_rumus = None
    loaded_model_belajar = None

    # Definisikan custom_objects di sini agar bisa digunakan
    custom_objects_for_rumus = {
        'sum_inputs_keras': sum_inputs_keras,
        'calculate_missing_digit_keras': calculate_missing_digit_keras,
        'calculate_index_keras': calculate_index_keras,
        'to_one_hot_keras': to_one_hot_keras
    }

    try:
        if os.path.exists(model_rumus_path):
            loaded_model_rumus = tf.keras.models.load_model(model_rumus_path, custom_objects=custom_objects_for_rumus)
            print(f"Model rumus eksplisit berhasil dimuat dari '{model_rumus_path}' dengan custom_objects.")
        else:
            print(f"File model rumus eksplisit tidak ditemukan di '{model_rumus_path}'.")

        if os.path.exists(model_belajar_path):
            # Model pembelajaran tidak menggunakan custom lambda layer jadi tidak perlu custom_objects
            loaded_model_belajar = tf.keras.models.load_model(model_belajar_path)
            print(f"Model pembelajaran berhasil dimuat dari '{model_belajar_path}'.")
        else:
            print(f"File model pembelajaran tidak ditemukan di '{model_belajar_path}'.")

        if loaded_model_rumus is not None and loaded_model_belajar is not None:
            test_models(loaded_model_rumus, loaded_model_belajar)
        elif loaded_model_rumus is not None and loaded_model_belajar is None:
            print("\nModel pembelajaran gagal dimuat. Hanya menguji model rumus.")
            test_single_model(loaded_model_rumus, "Rumus Eksplisit (setelah dimuat)")
        else:
            print("\nTidak dapat menguji model karena satu atau kedua model gagal dimuat.")

    except Exception as e:
        print(f"\nERROR menyeluruh saat memuat atau menguji model dari H5: {e}")
        print("Pastikan file H5 ada dan custom_objects sudah benar untuk model rumus eksplisit.")
        # Tambahkan traceback untuk debugging lebih lanjut jika error masih ada
        import traceback
        traceback.print_exc()


def test_single_model(model, model_name):
    print(f"\n--- Menguji Model: {model_name} ---")
    test_cases = [
        [2, 3, 4, 5, 6, 7, 8, 9], # Missing 1
        [1, 3, 4, 5, 6, 7, 8, 9], # Missing 2
        [1, 2, 4, 5, 6, 7, 8, 9], # Missing 3
        [1, 2, 3, 5, 6, 7, 8, 9], # Missing 4
        [1, 2, 3, 4, 6, 7, 8, 9], # Missing 5
        [1, 2, 3, 4, 5, 7, 8, 9], # Missing 6
        [1, 2, 3, 4, 5, 6, 8, 9], # Missing 7
        [1, 2, 3, 4, 5, 6, 7, 9], # Missing 8
        [1, 2, 3, 4, 5, 6, 7, 8]  # Missing 9
    ]
    X_test = np.array(test_cases, dtype=np.float32) # Pastikan tipe data float untuk input Keras

    predictions = model.predict(X_test)
    for i, pred_output in enumerate(predictions):
        missing_predicted_index = np.argmax(pred_output)
        missing_predicted_number = missing_predicted_index + 1
        if model_name.startswith("Pembelajaran"):
            prob = pred_output[missing_predicted_index]
            print(f"Input: {test_cases[i]}, Angka hilang diprediksi: {missing_predicted_number} (Prob: {prob:.4f})")
        else:
            print(f"Input: {test_cases[i]}, Angka hilang diprediksi: {missing_predicted_number}")

def test_models(model_rumus, model_belajar):
    print("\n--- Menguji kedua model (setelah dimuat dari H5) ---")
    test_single_model(model_rumus, "Rumus Eksplisit (setelah dimuat)")
    test_single_model(model_belajar, "Pembelajaran (setelah dimuat)")


if __name__ == "__main__":
    # Opsional: Atur seed untuk reproduktifitas jika diinginkan
    # np.random.seed(42)
    # tf.random.set_seed(42)

    save_and_test_models()

    print("\nProses eksekusi skrip selesai.")

    print("\nAnda bisa memuatnya di sesi lain menggunakan (pastikan fungsi helper juga didefinisikan atau diimpor):")
    print("import tensorflow as tf")
    print("from tensorflow.keras import backend as K")
    print("from nama_file_skrip import sum_inputs_keras, calculate_missing_digit_keras, calculate_index_keras, to_one_hot_keras")
    print("custom_objects_for_rumus = {")
    print("    'sum_inputs_keras': sum_inputs_keras,")
    print("    'calculate_missing_digit_keras': calculate_missing_digit_keras,")
    print("    'calculate_index_keras': calculate_index_keras,")
    print("    'to_one_hot_keras': to_one_hot_keras")
    print("}")
    print("loaded_model_rumus = tf.keras.models.load_model('models_h5/model_rumus_eksplisit.h5', custom_objects=custom_objects_for_rumus)")
    print("loaded_model_belajar = tf.keras.models.load_model('models_h5/model_pembelajaran.h5')")

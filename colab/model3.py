# Langkah 1: Impor Library yang dibutuhkan
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # Tambahkan untuk penskalaan data
import numpy as np
import os
import shutil
import json # Tambahkan untuk membaca file JSON
import sys # Tambahkan untuk keluar jika ada error fatal
from datetime import datetime # Tambahkan untuk mengelola tanggal

print("Langkah 1: Library yang dibutuhkan berhasil diimpor.")

# Path ke file data yang diunggah. Pastikan file ini ada di lingkungan Colab Anda.
# Anda dapat mengunggah file ini melalui panel File di sisi kiri Colab.
data_file_path = 'XAU_1d_data_processed.json'

# Langkah 2: Persiapan Data (Untuk Prediksi Harga dengan Tanggal)
print("\nLangkah 2: Memuat dan memproses data real dari", data_file_path, "untuk prediksi harga...")

# Memuat data dari file JSON dan memprosesnya
try:
    with open(data_file_path, 'r') as f:
        data = json.load(f)
    print("Data berhasil dimuat dari", data_file_path)

    # Memproses data:
    # Fitur (X): Kolom 1-5 (open, high, low, close, volume) untuk hari N
    # Label (y): Kolom 2, 3, 4 (high, low, close) untuk hari N+1 (harga hari berikutnya)
    # Tanggal: Kolom 0 untuk hari N (tanggal dari data fitur)
    # Mengabaikan baris header pertama jika ada ('NaT', null, ...)
    # Mengabaikan baris terakhir karena tidak ada data hari berikutnya untuk label

    valid_entries = []
    skipped_rows = 0
    # Iterasi hingga baris kedua terakhir, karena kita butuh data hari berikutnya untuk label
    for i in range(len(data) - 1):
        current_row = data[i]
        next_row = data[i+1]

        # Lewati baris pertama jika terlihat seperti header
        if i == 0 and isinstance(current_row, list) and len(current_row) > 0 and current_row[0] == "NaT":
            print("Melewati baris header pertama.")
            skipped_rows += 1
            continue

        # Cek jumlah kolom pada baris saat ini dan baris berikutnya
        if not isinstance(current_row, list) or len(current_row) < 7:
            # print(f"Melewati baris {i+1} karena format tidak valid atau jumlah kolom kurang: {current_row}") # Debugging opsional
            skipped_rows += 1
            continue
        if not isinstance(next_row, list) or len(next_row) < 7:
             # Baris berikutnya tidak valid, jadi baris saat ini tidak bisa digunakan sebagai fitur
             print(f"Melewati baris {i+1} karena baris berikutnya ({i+2}) tidak valid atau jumlah kolom kurang: {next_row}") # Debugging opsional
             skipped_rows += 1
             continue

        # Ambil tanggal dari baris saat ini
        date_str = current_row[0]
        # Coba parse tanggal untuk validasi (opsional tapi disarankan)
        try:
            # Mengasumsikan format ISO 8601 seperti "YYYY-MM-DDTHH:MM:SS"
            date_obj = datetime.fromisoformat(date_str)
            formatted_date = date_obj.strftime('%Y-%m-%d') # Format tanggal agar lebih mudah dibaca
        except ValueError:
            # print(f"Melewati baris {i+1} karena format tanggal tidak valid: '{date_str}'.") # Debugging opsional
            skipped_rows += 1
            continue

        # Cek apakah kolom fitur 1-5 pada baris saat ini adalah angka
        is_valid_features = True
        feature_values = []
        for j in range(1, 6): # Kolom 1 sampai 5 (open, high, low, close, volume)
            if not isinstance(current_row[j], (int, float)):
                # print(f"Melewati baris {i+1} karena fitur di kolom {j} ('{current_row[j]}') bukan angka.") # Debugging opsional
                is_valid_features = False
                break
            feature_values.append(float(current_row[j])) # Konversi ke float untuk konsistensi

        if not is_valid_features:
            skipped_rows += 1
            continue # Lanjut ke baris berikutnya jika fitur tidak valid

        # Cek apakah kolom label (kolom 2, 3, 4 - high, low, close) pada baris *berikutnya* adalah angka
        is_valid_labels = True
        label_values = []
        for j in [2, 3, 4]: # Kolom High (2), Low (3), Close (4) dari baris berikutnya
             if not isinstance(next_row[j], (int, float)):
                # print(f"Melewati baris {i+1} karena label di baris berikutnya ({i+2}), kolom {j} ('{next_row[j]}') bukan angka.") # Debugging opsional
                is_valid_labels = False
                break
             label_values.append(float(next_row[j])) # Konversi ke float untuk konsistensi

        if not is_valid_labels:
            skipped_rows += 1
            continue # Lanjut ke baris berikutnya jika label tidak valid


        # Jika baris saat ini dan baris berikutnya valid, tambahkan entri
        valid_entries.append({'date': formatted_date, 'features': feature_values, 'labels': label_values})


    print(f"Melewati {skipped_rows} baris yang tidak valid, merupakan header, atau tidak memiliki data label hari berikutnya.")

    if not valid_entries:
        raise ValueError("Tidak ada pasangan fitur-label yang valid ditemukan setelah pemrosesan. Pastikan file JSON berisi data dalam format yang diharapkan dan memiliki setidaknya 2 baris data valid.")

    # Memisahkan tanggal, fitur (X), dan label (y) dari valid_entries
    dates = np.array([item['date'] for item in valid_entries])
    X = np.array([item['features'] for item in valid_entries], dtype=np.float32) # Pastikan tipe data float32 untuk TensorFlow
    y = np.array([item['labels'] for item in valid_entries], dtype=np.float32) # Label juga float32 untuk regresi multi-output

    print("Data real berhasil diproses dan dikonversi.")
    print("Shape Tanggal:", dates.shape)
    print("Shape Fitur (X):", X.shape)
    print("Shape Label (y):", y.shape) # Sekarang seharusnya (jumlah_sampel, 3)

    # Penskalaan Fitur dan Label (penting untuk data finansial dan model NN)
    # Gunakan scaler terpisah untuk fitur dan label
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    print("Fitur data real berhasil diskalakan.")

    scaler_y = MinMaxScaler()
    # y sudah dalam bentuk 2D [n_samples, n_features (3)], jadi tidak perlu reshape
    y_scaled = scaler_y.fit_transform(y)
    print("Label data real berhasil diskalakan.")


    # Membagi data yang sudah diproses (tanggal, diskalakan) menjadi set pelatihan dan pengujian
    # Memastikan tanggal ikut terbagi bersama fitur dan label yang sesuai
    # Tidak menggunakan stratify karena ini tugas regresi
    dates_train, dates_test, X_train, X_test, y_train, y_test = train_test_split(
        dates, X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print("Data real dibagi:")
    print("dates_train={}, dates_test={}".format(dates_train.shape, dates_test.shape))
    print("X_train={}, X_test={}".format(X_train.shape, X_test.shape))
    print("y_train={}, y_test={}".format(y_train.shape, y_test.shape)) # Sekarang seharusnya (jumlah_sampel_split, 3)

except FileNotFoundError:
    print("ERROR FATAL: File data", data_file_path, "tidak ditemukan.")
    print("Pastikan file 'XAU_1d_data_processed.json' sudah diunggah ke sesi Colab Anda.")
    print("Skrip berhenti.")
    sys.exit(1) # Keluar dari skrip dengan kode error
except json.JSONDecodeError:
    print(f"ERROR FATAL: Gagal mem-parse file JSON '{data_file_path}'. Pastikan format file JSON valid.")
    print("Skrip berhenti.")
    sys.exit(1) # Keluar dari skrip dengan kode error
except Exception as e:
    print("ERROR FATAL saat memuat atau memproses data:", e)
    print("Skrip berhenti.")
    sys.exit(1) # Keluar dari skrip dengan kode error


# Definisikan SHUFFLE_BUFFER_SIZE (Dipertahankan untuk konsistensi, meskipun tidak digunakan langsung dengan tf.data.Dataset di sini)
SHUFFLE_BUFFER_SIZE = 100

# Langkah 3: Definisi Arsitektur Model (Disesuaikan untuk Regresi Multi-Output)
# Mempertahankan struktur lapisan tersembunyi, lapisan output 3 neuron untuk High, Low, Close
print("\nLangkah 3: Mendefinisikan arsitektur model untuk prediksi harga multi-output...")
# Input shape sesuai jumlah fitur data real setelah preprocessing (5 fitur)
input_dim = X_train.shape[1]
# Output shape 3 neuron untuk memprediksi High, Low, Close
output_dim = y_train.shape[1] # Akan otomatis 3

model = Sequential([
    tf.keras.layers.Input(shape=(input_dim,)), # Menggunakan Input Layer eksplisit
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    # Lapisan output untuk Regresi Multi-Output:
    # 3 unit karena memprediksi tiga nilai (High, Low, Close).
    # activation=None (linear activation) karena memprediksi nilai kontinu.
    Dense(output_dim, activation=None)
])
print("Arsitektur model dibuat.")
model.summary()


# Langkah 4: Konfigurasi Proses Pelatihan (Kompilasi Model - Disesuaikan untuk Regresi)
# Mengkonfigurasi model dengan optimizer, loss function, dan metrik evaluasi untuk regresi.
print("\nLangkah 4: Mengkompilasi model untuk prediksi harga multi-output...")
model.compile(optimizer='adam',
              loss='mean_squared_error', # MSE adalah loss umum untuk regresi
              metrics=['mean_absolute_error']) # MAE metrik umum, lebih mudah diinterpretasikan (error rata-rata)
print("Model dikompilasi (siap untuk pelatihan).")


# Langkah 5: Pelatihan Model
# Menggunakan parameter pelatihan dari eksekusi terakhir yang berhasil (21 epoch, batch size 89)
# Note: Hyperparameter ini mungkin perlu disetel ulang untuk tugas regresi
print("\nLangkah 5: Memulai pelatihan model selama 21 epoch dengan batch size 89...")
# Menambahkan validation_split untuk memantau kinerja pada sebagian data pelatihan
# Melatih menggunakan data fitur yang diskalakan (X_train) dan label yang diskalakan (y_train)
history = model.fit(X_train, y_train, epochs=21, batch_size=89, validation_split=0.2)
print("Pelatihan selesai.")


# Langkah 6: Evaluasi Model
print("\nLangkah 6: Mengevaluasi model pada data pengujian...")
# Mengevaluasi menggunakan data fitur yang diskalakan (X_test) dan label yang diskalakan (y_test)
loss, mae = model.evaluate(X_test, y_test, verbose=0) # Metrik sekarang MAE, bukan akurasi
print("Hasil Evaluasi pada data pengujian (nilai diskalakan): Loss (MSE) = {:.6f}, MAE = {:.6f}".format(loss, mae))

# Untuk mendapatkan MAE dalam skala harga asli, kita perlu melakukan prediksi pada X_test,
# meng-inverse-scale hasilnya, dan menghitung MAE terhadap y_test (dalam skala asli).
# Ini dilakukan di Langkah 7.


# Langkah 7: Membuat prediksi pada data baru (dengan Tanggal)
print("\nLangkah 7: Membuat prediksi pada data baru...")
# Mengambil beberapa sampel data real terbaru dari set pengujian untuk prediksi
# Menggunakan 5 sampel terakhir dari set pengujian (fitur diskalakan)
num_samples_to_predict = 5
X_new_scaled = X_test[-num_samples_to_predict:]
dates_new = dates_test[-num_samples_to_predict:] # Ambil tanggal yang sesuai

# Lakukan prediksi pada data baru yang diskalakan
predictions_scaled = model.predict(X_new_scaled)

print("Data baru untuk prediksi (fitur setelah penskalaan) dan Tanggal:")
for i in range(num_samples_to_predict):
    print(f"Tanggal: {dates_new[i]}, Fitur Skalakan: {X_new_scaled[i]}")


print("\nHasil Prediksi (harga output model, masih diskalakan) dan Tanggal:")
for i in range(num_samples_to_predict):
    print(f"Tanggal: {dates_new[i]}, Prediksi Skalakan: {predictions_scaled[i]}")


# Meng-inverse-scale prediksi untuk mendapatkan perkiraan harga dalam skala asli
predictions_original_scale = scaler_y.inverse_transform(predictions_scaled)
print("\nHasil Prediksi (harga dalam skala asli) dan Tanggal:")
print("Tanggal       | Prediksi High | Prediksi Low  | Prediksi Close")
print("---------------------------------------------------------------")
for i in range(num_samples_to_predict):
    print(f"{dates_new[i]} | {predictions_original_scale[i, 0]:<13.4f} | {predictions_original_scale[i, 1]:<13.4f} | {predictions_original_scale[i, 2]:<14.4f}")


# Untuk perbandingan, mari lihat harga High, Low, Close asli dari 5 sampel terakhir di set pengujian (y_test)
# Kita perlu meng-inverse-scale y_test[-5:] untuk melihat nilai aslinya
y_new_original_scale = scaler_y.inverse_transform(y_test[-num_samples_to_predict:])
print("\nHarga High, Low, Close asli untuk 5 sampel terakhir di set pengujian:")
print("Tanggal       | Asli High     | Asli Low      | Asli Close")
print("-------------------------------------------------------------")
for i in range(num_samples_to_predict):
     # Cari tanggal yang sesuai di dates_test
     current_date = dates_test[-num_samples_to_predict:][i]
     print(f"{current_date} | {y_new_original_scale[i, 0]:<13.4f} | {y_new_original_scale[i, 1]:<13.4f} | {y_new_original_scale[i, 2]:<14.4f}")


# Hitung MAE untuk 5 sampel prediksi ini (dalam skala asli)
# MAE dihitung untuk setiap output (High, Low, Close) dan kemudian dirata-ratakan
mae_on_new_samples = np.mean(np.abs(predictions_original_scale - y_new_original_scale))
print(f"\nMAE rata-rata pada 5 sampel prediksi baru (skala asli): {mae_on_new_samples:.4f}")


# Langkah 8: Menyimpan Model
print("\nLangkah 8: Menyimpan model...")

# Nama file dan direktori untuk penyimpanan
h5_model_path = 'price_prediction_model.h5' # Ganti nama file agar lebih deskriptif
savedmodel_dir = 'price_prediction_savedmodel' # Ganti nama direktori
keras_model_path = 'price_prediction_model.keras' # Format native Keras yang baru

# Menyimpan dalam format HDF5 (.h5)
try:
    model.save(h5_model_path)
    print("Model berhasil disimpan dalam format HDF5 ke '{}'".format(h5_model_path))
except Exception as e:
    print("Gagal menyimpan model dalam format HDF5:", e)

# Menyimpan dalam format SavedModel (.export())
try:
    # Hapus direktori SavedModel jika sudah ada (untuk menghindari error)
    if os.path.exists(savedmodel_dir):
        shutil.rmtree(savedmodel_dir)
        print("Direktori '{}' berhasil dihapus sebelum disimpan.".format(savedmodel_dir))

    model.export(savedmodel_dir) # Menggunakan .export() untuk SavedModel
    print("Model berhasil diekspor dalam format SavedModel ke '{}'".format(savedmodel_dir))
except Exception as e:
    print("Gagal mengekspor model dalam format SavedModel:", e)


# Menyimpan dalam format Native Keras (.keras)
try:
    model.save(keras_model_path) # Menggunakan .save() dengan ekstensi .keras
    print("Model berhasil disimpan dalam format Native Keras ke '{}'".format(keras_model_path))
except Exception as e:
    print("Gagal menyimpan model dalam format Native Keras:", e)


# Langkah 9: Memuat kembali model yang sudah disimpan
print("\nLangkah 9: Memuat kembali model yang sudah disimpan...")

loaded_model_h5 = None
loaded_model_keras = None

# Memuat dari format HDF5
if os.path.exists(h5_model_path):
    try:
        loaded_model_h5 = tf.keras.models.load_model(h5_model_path)
        print("Model berhasil dimuat dari '{}' (HDF5)".format(h5_model_path))
    except Exception as e:
        print("Gagal memuat model dari '{}' (HDF5): {}".format(h5_model_path, e))
else:
    print(f"File '{h5_model_path}' tidak ditemukan, tidak bisa memuat HDF5 model.")


# Memuat dari format SavedModel (akan tetap gagal di Keras 3 menggunakan tf.keras.models.load_model)
# Baris ini sengaja dipertahankan untuk menunjukkan perilaku ini seperti output sebelumnya.
# Untuk memuat SavedModel yang diekspor dengan `.export()` di Keras 3
# memerlukan cara yang berbeda (misalnya tf.saved_model.load atau TFSMLayer)
# tf.keras.models.load_model() tidak mendukungnya secara langsung.
if os.path.exists(savedmodel_dir):
    try:
        # Ini akan gagal di Keras 3 untuk SavedModel yang diekspor dengan .export()
        loaded_model_savedmodel = tf.keras.models.load_model(savedmodel_dir)
        print("Model berhasil dimuat dari '{}' (SavedModel) menggunakan tf.keras.models.load_model".format(savedmodel_dir))
    except Exception as e:
        print("Gagal memuat model dari '{}' (SavedModel) menggunakan tf.keras.models.load_model: {}".format(savedmodel_dir, e))
else:
    print(f"Direktori '{savedmodel_dir}' tidak ditemukan, tidak bisa memuat SavedModel.")


# Memuat dari format Native Keras
if os.path.exists(keras_model_path):
    try:
        loaded_model_keras = tf.keras.models.load_model(keras_model_path)
        print("Model berhasil dimuat dari '{}' (Native Keras)".format(keras_model_path))
    except Exception as e:
        print("Gagal memuat model dari '{}' (Native Keras): {}".format(keras_model_path, e))
else:
    print(f"File '{keras_model_path}' tidak ditemukan, tidak bisa memuat Native Keras model.")


# Langkah 10: Menggunakan model yang dimuat untuk prediksi pada data baru (dengan Tanggal)
print("\nLangkah 10: Menggunakan model yang dimuat untuk prediksi pada data baru...")

# Gunakan data baru yang sama (X_new_scaled) dan tanggal yang sesuai (dates_new)
if loaded_model_h5 is not None:
    try:
        predictions_h5_scaled = loaded_model_h5.predict(X_new_scaled)
        predictions_h5_original_scale = scaler_y.inverse_transform(predictions_h5_scaled)
        print("Prediksi dari model yang dimuat (HDF5, skala asli) dan Tanggal:")
        print("Tanggal       | Prediksi High | Prediksi Low  | Prediksi Close")
        print("---------------------------------------------------------------")
        for i in range(num_samples_to_predict):
            print(f"{dates_new[i]} | {predictions_h5_original_scale[i, 0]:<13.4f} | {predictions_h5_original_scale[i, 1]:<13.4f} | {predictions_h5_original_scale[i, 2]:<14.4f}")

        # Verifikasi apakah prediksi cocok dengan model asli
        # Bandingkan prediksi skala asli dengan prediksi asli dari model yang dilatih
        if np.allclose(predictions_original_scale, predictions_h5_original_scale):
            print("Verifikasi: Prediksi dari model asli dan model HDF5 yang dimuat cocok.")
        else:
            print("Verifikasi: Prediksi dari model asli dan model HDF5 yang dimuat TIDAK cocok.")
    except Exception as e:
         print("Gagal melakukan prediksi dengan model HDF5 yang dimuat:", e)


if loaded_model_keras is not None:
    try:
        predictions_keras_scaled = loaded_model_keras.predict(X_new_scaled)
        predictions_keras_original_scale = scaler_y.inverse_transform(predictions_keras_scaled)
        print("\nPrediksi dari model yang dimuat (Native Keras, skala asli) dan Tanggal:")
        print("Tanggal       | Prediksi High | Prediksi Low  | Prediksi Close")
        print("---------------------------------------------------------------")
        for i in range(num_samples_to_predict):
            print(f"{dates_new[i]} | {predictions_keras_original_scale[i, 0]:<13.4f} | {predictions_keras_original_scale[i, 1]:<13.4f} | {predictions_keras_original_scale[i, 2]:<14.4f}")
        # Verifikasi apakah prediksi cocok dengan model asli
         # Bandingkan prediksi skala asli dengan prediksi asli dari model yang dilatih
        if np.allclose(predictions_original_scale, predictions_keras_original_scale):
            print("Verifikasi: Prediksi dari model asli dan model Native Keras yang dimuat cocok.")
        else:
            print("Verifikasi: Prediksi dari model asli dan model Native Keras yang dimuat TIDAK cocok.")
    except Exception as e:
         print("Gagal melakukan prediksi dengan model Native Keras yang dimuat:", e)


print("\nPipeline AI selesai dieksekusi dengan data real untuk prediksi harga.")

# --- Bagian 1: Impor Library ---
# Impor modul dan kelas yang dibutuhkan dari library TensorFlow dan Keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential # Import kelas Sequential untuk model tumpukan
from tensorflow.keras.layers import Dense # Import kelas Dense untuk lapisan terhubung penuh
from sklearn.model_selection import train_test_split # Import fungsi untuk membagi data (dari scikit-learn)
import os # Import modul os untuk operasi sistem file
import shutil # Import modul shutil untuk operasi file/direktori tingkat tinggi

print("Langkah 1: Library berhasil diimpor.")

# --- Bagian 2: Persiapan Data ---
# Membuat data sintetis (dummy) untuk contoh ini.
# Dalam proyek nyata, data akan dimuat dari file (CSV, gambar, dll.)
# dan mungkin diproses lebih lanjut menggunakan tf.data API.

jumlah_sampel = 1000 # Jumlah total data sampel
jumlah_fitur = 10    # Jumlah fitur per sampel

# Membuat fitur (X) menggunakan NumPy: angka acak antara 0 dan 1
X = np.random.rand(jumlah_sampel, jumlah_fitur).astype(np.float32)

# Membuat label (y) menggunakan NumPy: klasifikasi biner (0 atau 1)
# Logika dummy: label 1 jika jumlah 5 fitur pertama > 2.5, jika tidak 0
y = (np.sum(X[:, :5], axis=1) > 2.5).astype(np.int32)

print(f"\nLangkah 2: Data dummy dibuat dengan shape X={X.shape}, y={y.shape}")

# Membagi data menjadi set pelatihan (training) dan pengujian (testing)
# Data pelatihan digunakan model untuk belajar. Data pengujian untuk mengevaluasi kinerja akhir.
# test_size=0.2 berarti 20% data untuk pengujian, 80% untuk pelatihan.
# random_state=42 untuk hasil pembagian yang konsisten setiap kali dijalankan.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data dibagi: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

# Definisikan SHUFFLE_BUFFER_SIZE (Diperlukan jika menggunakan tf.data.Dataset dengan shuffle, meskipun tidak digunakan eksplisit di sini, baik untuk didefinisikan)
SHUFFLE_BUFFER_SIZE = 100

# --- Bagian 3: Definisi Arsitektur Model ---
# Mendefinisikan struktur jaringan saraf menggunakan Keras Sequential API.
# Sequential model adalah tumpukan lapisan, cocok untuk arsitektur sederhana.

model = Sequential ([
    # Lapisan input dan lapisan tersembunyi pertama (Dense)
    # 32 adalah jumlah neuron di lapisan ini.
    # activation='relu' adalah fungsi aktivasi Rectified Linear Unit.
    # input_shape=(jumlah_fitur,) menentukan bentuk input yang diharapkan (tanpa batch size).
    tf.keras.layers.Input(shape=(jumlah_fitur,)), # Menggunakan Input Layer eksplisit (opsional tapi baik)
    Dense(32, activation='relu'),

    # Lapisan tersembunyi kedua (Dense)
    Dense(16, activation='relu'),

    # Lapisan output (Dense)
    # 1 unit karena ini tugas klasifikasi biner (satu output probabilitas).
    # activation='sigmoid' menghasilkan output antara 0 dan 1 (probabilitas).
    Dense(1, activation='sigmoid')
])

print("\nLangkah 3: Arsitektur model dibuat.")
model.summary() # Menampilkan ringkasan arsitektur model (jumlah parameter, shape output setiap layer)

# --- Bagian 4: Konfigurasi Proses Pelatihan (Kompilasi Model) ---
# Mengkonfigurasi model dengan menentukan optimizer, loss function, dan metrik evaluasi.
# Optimizer: Algoritma yang digunakan untuk menyesuaikan bobot model. 'adam' adalah pilihan populer.
# Loss: Fungsi yang mengukur seberapa 'salah' prediksi model. 'binary_crossentropy' untuk klasifikasi biner.
# Metrics: Metrik untuk memantau kinerja model selama pelatihan dan evaluasi. 'accuracy' adalah metrik umum.

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nLangkah 4: Model dikompilasi (siap untuk pelatihan).")

# --- Bagian 5: Pelatihan Model ---
# Model belajar dari data pelatihan dengan melakukan forward dan backward pass.

EPOCHS = 21 # Jumlah kali model akan melihat seluruh data pelatihan
BATCH_SIZE = 89 # Jumlah sampel data yang diproses per langkah pelatihan

print(f"\nLangkah 5: Memulai pelatihan model selama {EPOCHS} epoch dengan batch size {BATCH_SIZE}...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.618 # Menggunakan 20% dari data pelatihan sebagai data validasi
    # Data validasi digunakan untuk memantau kinerja model pada data yang tidak digunakan untuk update bobot
)

print("\nPelatihan selesai.")

# --- Bagian 6: Evaluasi Model ---
# Mengukur kinerja model yang sudah dilatih pada data pengujian (yang belum pernah dilihat model).

print("\nLangkah 6: Mengevaluasi model pada data pengujian...")
# model.evaluate mengembalikan nilai loss dan metrik yang dikonfigurasi saat compile.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0) # verbose=0 agar tidak mencetak progress bar

print(f"Hasil Evaluasi pada data pengujian: Loss = {loss:.4f}, Akurasi = {accuracy:.4f}")

# --- Bagian 7: Prediksi ---
# Menggunakan model yang sudah dilatih untuk membuat prediksi pada data baru.

# Membuat data baru dummy untuk prediksi (misal 5 sampel baru)
X_new = np.random.rand(5, jumlah_fitur).astype(np.float32)

print("\nLangkah 7: Membuat prediksi pada data baru...")
# model.predict menghasilkan output mentah dari lapisan output (probabilitas jika sigmoid).
predictions_prob = model.predict(X_new)

# Untuk tugas klasifikasi biner dengan aktivasi sigmoid, output adalah probabilitas.
# Kita bisa mengubah probabilitas menjadi kelas biner (0 atau 1) dengan threshold (misal 0.5).
predictions_class = (predictions_prob > 0.5).astype(np.int32)

print("Data baru untuk prediksi:\n", X_new)
print("\nHasil Prediksi (Probabilitas output model):\n", predictions_prob)
print("\nHasil Prediksi (Kelas biner setelah threshold 0.5):\n", predictions_class)

# --- Bagian 8: Menyimpan Model ---
# Menyimpan model yang sudah dilatih agar bisa digunakan nanti.

# Nama file/direktori untuk model
model_h5_path = 'simple_pipeline_model.h5'
model_savedmodel_dir = 'simple_pipeline_savedmodel'
model_keras_path = 'simple_pipeline_model.keras' # Format native Keras yang baru

print(f"\nLangkah 8: Menyimpan model...")

# Menyimpan dalam format HDF5 (cara lama Keras, cocok untuk model Keras sederhana)
try:
    model.save(model_h5_path)
    print(f"Model berhasil disimpan dalam format HDF5 ke '{model_h5_path}'")
except Exception as e:
    print(f"Gagal menyimpan model dalam format HDF5: {e}")

# Menyimpan dalam format SavedModel (direkomendasikan oleh TensorFlow untuk deployment)
# Gunakan model.export() untuk SavedModel
if os.path.exists(model_savedmodel_dir):
    try:
        shutil.rmtree(model_savedmodel_dir) # Hapus direktori jika sudah ada
        print(f"Direktori '{model_savedmodel_dir}' berhasil dihapus sebelum disimpan.")
    except Exception as e:
        print(f"Gagal menghapus direktori '{model_savedmodel_dir}': {e}")

try:
    # Menggunakan model.export() untuk SavedModel
    model.export(model_savedmodel_dir)
    print(f"Model berhasil diekspor dalam format SavedModel ke '{model_savedmodel_dir}'")
except Exception as e:
     print(f"Gagal mengekspor model dalam format SavedModel: {e}")

# Menyimpan dalam format Native Keras (.keras) (format rekomendasi baru untuk Keras)
try:
    model.save(model_keras_path)
    print(f"Model berhasil disimpan dalam format Native Keras ke '{model_keras_path}'")
except Exception as e:
    print(f"Gagal menyimpan model dalam format Native Keras: {e}")


# --- Bagian 9: Memuat Model ---
# Memuat kembali model yang sudah disimpan.

print("\nLangkah 9: Memuat kembali model yang sudah disimpan...")

loaded_model_h5 = None
loaded_model_savedmodel_keras = None
loaded_model_keras = None

# Memuat model dari format HDF5
if os.path.exists(model_h5_path):
    try:
        loaded_model_h5 = tf.keras.models.load_model(model_h5_path)
        print(f"Model berhasil dimuat dari '{model_h5_path}' (HDF5)")
    except Exception as e:
        print(f"Gagal memuat model dari '{model_h5_path}' (HDF5): {e}")
else:
    print(f"File '{model_h5_path}' tidak ditemukan, tidak bisa memuat HDF5 model.")


# Memuat model dari format SavedModel (Direkomendasikan menggunakan tf.keras.models.load_model)
if os.path.exists(model_savedmodel_dir):
    try:
        loaded_model_savedmodel_keras = tf.keras.models.load_model(model_savedmodel_dir)
        print(f"Model berhasil dimuat dari '{model_savedmodel_dir}' menggunakan tf.keras.models.load_model (SavedModel)")
    except Exception as e:
        print(f"Gagal memuat model dari '{model_savedmodel_dir}' (SavedModel) menggunakan tf.keras.models.load_model: {e}")
    # Jika model aslinya bukan model Keras (misal, hanya tf.function), bisa pakai tf.saved_model.load
    # loaded_model_savedmodel_tf = tf.saved_model.load(model_savedmodel_dir)
    # print(f"Model berhasil dimuat dari '{model_savedmodel_dir}' menggunakan tf.saved_model.load (SavedModel)")
else:
    print(f"Direktori '{model_savedmodel_dir}' tidak ditemukan, tidak bisa memuat SavedModel.")

# Memuat model dari format Native Keras (.keras)
if os.path.exists(model_keras_path):
    try:
        loaded_model_keras = tf.keras.models.load_model(model_keras_path)
        print(f"Model berhasil dimuat dari '{model_keras_path}' (Native Keras)")
    except Exception as e:
        print(f"Gagal memuat model dari '{model_keras_path}' (Native Keras): {e}")
else:
    print(f"File '{model_keras_path}' tidak ditemukan, tidak bisa memuat Native Keras model.")


# --- Bagian 10: Menggunakan Model yang Dimuat untuk Prediksi ---
# Memverifikasi bahwa model yang dimuat berfungsi dengan benar.

print("\nLangkah 10: Menggunakan model yang dimuat untuk prediksi pada data baru...")

# Menggunakan model yang dimuat dari HDF5 (jika berhasil dimuat)
if loaded_model_h5 is not None:
    try:
        predictions_loaded_h5 = loaded_model_h5.predict(X_new)
        print("Prediksi dari model yang dimuat (HDF5):\n", predictions_loaded_h5)
        # Periksa apakah prediksi dari model asli dan model yang dimuat sama
        np.testing.assert_allclose(predictions_prob, predictions_loaded_h5, rtol=1e-6)
        print("Verifikasi: Prediksi dari model asli dan model HDF5 yang dimuat cocok.")
    except Exception as e:
        print(f"Gagal melakukan prediksi dengan model HDF5 yang dimuat: {e}")


# Menggunakan model yang dimuat dari SavedModel (via Keras load, jika berhasil dimuat)
if loaded_model_savedmodel_keras is not None:
    try:
        predictions_loaded_savedmodel = loaded_model_savedmodel_keras.predict(X_new)
        print("Prediksi dari model yang dimuat (SavedModel via Keras load):\n", predictions_loaded_savedmodel)
         # Periksa apakah prediksi dari model asli dan model yang dimuat sama
        np.testing.assert_allclose(predictions_prob, predictions_loaded_savedmodel, rtol=1e-6)
        print("Verifikasi: Prediksi dari model asli dan model SavedModel yang dimuat cocok.")
    except Exception as e:
        print(f"Gagal melakukan prediksi dengan model SavedModel yang dimuat: {e}")

# Menggunakan model yang dimuat dari Native Keras (jika berhasil dimuat)
if loaded_model_keras is not None:
    try:
        predictions_loaded_keras = loaded_model_keras.predict(X_new)
        print("Prediksi dari model yang dimuat (Native Keras):\n", predictions_loaded_keras)
         # Periksa apakah prediksi dari model asli dan model yang dimuat sama
        np.testing.assert_allclose(predictions_prob, predictions_loaded_keras, rtol=1e-6)
        print("Verifikasi: Prediksi dari model asli dan model Native Keras yang dimuat cocok.")
    except Exception as e:
        print(f"Gagal melakukan prediksi dengan model Native Keras yang dimuat: {e}")


# --- Opsional: Membersihkan file model ---
# print("\nMembersihkan file model...")
# if os.path.exists(model_h5_path):
#     try:
#         os.remove(model_h5_path)
#         print(f"File '{model_h5_path}' berhasil dihapus.")
#     except Exception as e:
#         print(f"Gagal menghapus file '{model_h5_path}': {e}")
#
# if os.path.exists(model_savedmodel_dir):
#     try:
#         shutil.rmtree(model_savedmodel_dir)
#         print(f"Direktori '{model_savedmodel_dir}' berhasil dihapus.")
#     except Exception as e:
#         print(f"Gagal menghapus direktori '{model_savedmodel_dir}': {e}")
#
# if os.path.exists(model_keras_path):
#     try:
#         os.remove(model_keras_path)
#         print(f"File '{model_keras_path}' berhasil dihapus.")
#     except Exception as e:
#         print(f"Gagal menghapus file '{model_keras_path}': {e}")

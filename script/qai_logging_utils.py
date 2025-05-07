import logging
import os
import sys
from datetime import datetime

# --- Konfigurasi Dasar Logging ---
# Anda bisa memindahkan konfigurasi ini ke quantai_confiq.yaml jika ingin lebih fleksibel
LOG_LEVEL = logging.INFO # Tingkat log default (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_DIR = 'results/logs' # Direktori untuk menyimpan file log
LOG_FILE_TEMPLATE = '{stage}_{timestamp}.log' # Template nama file log

# Pastikan direktori log ada
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging(log_level=LOG_LEVEL, log_to_file=True, stage_name="general"):
    """
    Mengatur konfigurasi logging untuk script.
    Log akan ditampilkan di konsol dan opsional disimpan ke file.
    """
    # Dapatkan logger root
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Hapus handler yang sudah ada untuk menghindari duplikasi log
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format untuk log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler untuk Konsol
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler untuk File Log (opsional)
    if log_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_name = LOG_FILE_TEMPLATE.format(stage=stage_name, timestamp=timestamp)
        log_file_path = os.path.join(LOG_DIR, log_file_name)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging diatur. Log juga ditulis ke file: {log_file_path}")

    # Mengembalikan logger root
    return logger

# --- Contoh Penggunaan (di script lain) ---
# Di awal script lain (misal: qai_data_prep.py):
# from scripts.qai_logging_utils import setup_logging
# logger = setup_logging(stage_name="data_prep") # Atur logging untuk tahap 'data_prep'

# Kemudian di dalam script, gunakan logger:
# logger.info("Memulai proses persiapan data...")
# logger.warning("Ada nilai null yang ditemukan.")
# logger.error("Gagal memuat file data.")
# logger.debug("Variabel X_train shape: %s", X_train.shape) # Gunakan placeholder %s, %d, dll.

# --- Contoh Penggunaan di script ini (jika dijalankan langsung) ---
if __name__ == "__main__":
    # Atur logging untuk script ini
    logger = setup_logging(log_level=logging.DEBUG, stage_name="logging_test")

    # Contoh pesan log
    logger.info("Ini adalah pesan informasi.")
    logger.warning("Ini adalah pesan peringatan.")
    logger.error("Ini adalah pesan kesalahan.")
    logger.debug("Ini adalah pesan debug (hanya terlihat jika log_level <= DEBUG).")

    print("\nScript logging_utils selesai. Cek output konsol dan folder 'results/logs/'.")


import yaml
import os

# --- Konfigurasi Path File Konfigurasi ---
# Sesuaikan path ini agar sesuai dengan lokasi di repository GitHub Anda
# Jika Anda menjalankan script ini dari root folder 'my_quantai_project'
CONFIG_FILE_PATH = 'config/project_config.yaml'

class ConfigManager:
    """
    Kelas untuk memuat dan mengakses konfigurasi proyek dari file YAML.
    """
    def __init__(self, config_path=CONFIG_FILE_PATH):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        """Memuat konfigurasi dari file YAML."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan di: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"Konfigurasi berhasil dimuat dari {config_path}")
            return config_data
        except yaml.YAMLError as e:
            print(f"ERROR saat memparsing file YAML: {e}")
            raise # Re-raise exception
        except Exception as e:
            print(f"ERROR saat memuat file konfigurasi: {e}")
            raise # Re-raise exception

    def get(self, key, default=None):
        """
        Mendapatkan nilai konfigurasi menggunakan kunci (misal: 'data.window_size').
        Mendukung akses bersarang menggunakan dot notation.
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                print(f"Peringatan: Kunci konfigurasi '{key}' tidak ditemukan. Menggunakan nilai default: {default}")
                return default
            else:
                print(f"ERROR: Kunci konfigurasi '{key}' tidak ditemukan dan tidak ada nilai default.")
                raise KeyError(f"Kunci konfigurasi '{key}' tidak ditemukan.")

# --- Contoh Penggunaan ---
if __name__ == "__main__":
    try:
        # Inisialisasi ConfigManager
        config_manager = ConfigManager()

        # Akses beberapa konfigurasi
        window_size = config_manager.get('data.window_size')
        epochs = config_manager.get('train.epochs')
        model_name = config_manager.get('model.name')
        optimizer_lr = config_manager.get('train.optimizer.config.lr')
        scaler_target_path = config_manager.get('data.scaler_target_path')

        print("\n--- Contoh Nilai Konfigurasi yang Diakses ---")
        print(f"Window Size: {window_size}")
        print(f"Epochs: {epochs}")
        print(f"Nama Model: {model_name}")
        print(f"Learning Rate Optimizer: {optimizer_lr}")
        print(f"Path Scaler Target: {scaler_target_path}")

        # Contoh akses kunci yang tidak ada (dengan default)
        non_existent_key = config_manager.get('some.other.setting', default='default_value')
        print(f"Kunci tidak ada (dengan default): {non_existent_key}")

        # Contoh akses kunci yang tidak ada (tanpa default - akan raise error)
        # try:
        #     will_fail = config_manager.get('non.existent.key.without.default')
        # except KeyError as e:
        #     print(f"\nBerhasil menangani error: {e}")


    except FileNotFoundError:
        print("Pastikan file 'config/project_config.yaml' ada di lokasi yang benar.")
    except Exception as e:
        print(f"Terjadi error: {e}")


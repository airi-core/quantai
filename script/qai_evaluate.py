import tensorflow as tf
import numpy as np
import joblib # Library untuk memuat scaler
import os
import sys

# Add the parent directory to the system path to allow importing modules from 'scripts'
# This assumes you are running the script from the root directory of the project
# If running from 'scripts' folder, you might need to adjust this path manipulation
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
sys.path.append(project_root_dir)

# Import necessary components from other scripts
from scripts.qai_config_manager import ConfigManager

# We will need functions/logic to load the prepared data (X_test, y_test)
# Ideally, the data preparation script would save these arrays, or there's a dedicated
# data loading function that can load the split data using the config.
# For this script, we'll assume the test data arrays (.npy files) are saved.
# You would need to add code in qai_data_prep.py to save these arrays.
# Example:
# np.save('data/processed/X_test.npy', X_test)
# np.save('data/processed/y_test.npy', y_test)
# Also, ensure the paths to these files are in your config/quantai_confiq.yaml.

# --- Main Evaluation Script ---
def main():
    """
    Main function to load a saved model and evaluate it on test data.
    """
    # --- 1. Load Configuration ---
    # Using ConfigManager to get all necessary paths and parameters
    try:
        config_manager = ConfigManager() # Loads from default path 'config/quantai_confiq.yaml'
        config = config_manager.config

        # Get paths from config
        model_path = config_manager.get('save.model_save_path') # Path where the model was saved
        scaler_target_path = config_manager.get('data.scaler_target_path') # Path to the target scaler

        # Assuming X_test and y_test were saved as .npy files during data prep
        # Add these paths to your config/quantai_confiq.yaml under 'data' section
        # Example config entry:
        # data:
        #   ...
        #   X_test_path: data/processed/X_test.npy
        #   y_test_path: data/processed/y_test.npy
        X_test_path = config_manager.get('data.X_test_path')
        y_test_path = config_manager.get('data.y_test_path')

    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR loading configuration or required paths: {e}")
        print("Please ensure 'config/quantai_confiq.yaml' exists and contains paths for model, scalers, and test data.")
        sys.exit(1) # Exit if config or paths are missing
    except Exception as e:
        print(f"An unexpected error occurred during config loading: {e}")
        sys.exit(1)


    print(f"--- Starting Model Evaluation ---")

    # --- 2. Load Model and Scaler ---
    try:
        print(f"Memuat model dari: {model_path}")
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model berhasil dimuat.")

        print(f"Memuat scaler target dari: {scaler_target_path}")
        loaded_scaler_target = joblib.load(scaler_target_path)
        print("Scaler target berhasil dimuat.")

    except FileNotFoundError as e:
        print(f"ERROR: File model atau scaler tidak ditemukan - {e}.")
        print("Pastikan path dalam konfigurasi benar dan file-file tersebut ada.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR saat memuat model atau scaler: {e}")
        sys.exit(1)

    # --- 3. Load Test Data ---
    try:
        print(f"Memuat data pengujian dari: {X_test_path} dan {y_test_path}")
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        print(f"Data pengujian berhasil dimuat. Bentuk X_test: {X_test.shape}, y_test: {y_test.shape}")

    except FileNotFoundError as e:
        print(f"ERROR: File data pengujian tidak ditemukan - {e}.")
        print("Pastikan path dalam konfigurasi benar dan file-file .npy data pengujian ada.")
        print("Anda mungkin perlu memodifikasi qai_data_prep.py untuk menyimpan X_test dan y_test.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR saat memuat data pengujian: {e}")
        sys.exit(1)


    # --- 4. Evaluate Model on Scaled Data ---
    print("\nMengevaluasi model pada data pengujian (scaled)...")
    # Use the same metrics as during training
    loss_scaled, mae_scaled, mse_scaled = loaded_model.evaluate(X_test, y_test, verbose=0)

    print(f"Hasil Evaluasi pada Data Pengujian (Scaled):")
    print(f"  Loss (MSE): {loss_scaled:.6f}")
    print(f"  MAE: {mae_scaled:.6f}")
    print(f"  MSE: {mse_scaled:.6f}")

    # --- 5. Evaluate on Original Scale (Inverse Scaling) ---
    print("\nMengevaluasi model pada skala harga asli (Inverse Scaling)...")

    # Lakukan prediksi pada data pengujian
    y_pred_scaled = loaded_model.predict(X_test)

    # Lakukan inverse scaling pada prediksi dan nilai y_test asli
    # Gunakan scaler_target yang sudah dilatih pada data HLC mentah
    try:
        y_test_original_scale = loaded_scaler_target.inverse_transform(y_test)
        y_pred_original_scale = loaded_scaler_target.inverse_transform(y_pred_scaled)

        # Hitung metrik (MAE dan MSE) pada skala harga asli
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae_original_scale = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
        mse_original_scale = mean_squared_error(y_test_original_scale, y_pred_original_scale)
        rmse_original_scale = np.sqrt(mse_original_scale) # RMSE seringkali lebih mudah diinterpretasikan

        print(f"Hasil Evaluasi pada Data Pengujian (Skala Harga Asli):")
        print(f"  MAE: {mae_original_scale:.4f}")
        print(f"  MSE: {mse_original_scale:.4f}")
        print(f"  RMSE: {rmse_original_scale:.4f}")

        # --- Opsional: Simpan hasil metrik ke file ---
        # results_dir = 'results/evaluation_metrics' # Define results directory
        # os.makedirs(results_dir, exist_ok=True)
        # results_file = os.path.join(results_dir, 'test_evaluation_metrics.json')
        # evaluation_results = {
        #     'mae_original_scale': mae_original_scale,
        #     'mse_original_scale': mse_original_scale,
        #     'rmse_original_scale': rmse_original_scale,
        #     'mae_scaled': float(mae_scaled), # Convert numpy float to Python float for JSON
        #     'mse_scaled': float(mse_scaled),
        #     'loss_scaled': float(loss_scaled)
        # }
        # with open(results_file, 'w') as f:
        #     json.dump(evaluation_results, f, indent=4)
        # print(f"\nHasil evaluasi disimpan ke: {results_file}")


    except Exception as e:
        print(f"ERROR saat melakukan inverse scaling atau menghitung metrik asli: {e}")
        sys.exit(1)


# This ensures the main function is called when the script is executed directly
if __name__ == "__main__":
    main()

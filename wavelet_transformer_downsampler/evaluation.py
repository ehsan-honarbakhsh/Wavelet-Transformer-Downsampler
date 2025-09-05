import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy.interpolate import interp1d, CubicSpline
import os
import argparse
import logging
from wavelet_transformer import WaveletDownsamplingModel, TimeSeriesEmbedding, DownsampleTransformerBlock, get_wavedec_coeff_lengths, downsampling_loss


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log', mode='w', delay=True)
    ]
)
logger = logging.getLogger(__name__)


def load_m4_daily(test_file_path, max_length=150):

    if not os.path.exists(test_file_path):
        raise FileNotFoundError(
            f"M4 Daily test dataset file not found at: {test_file_path}")

    test_df_test = pd.read_csv(test_file_path)
    test_df = test_df_test[22000:]
    X_test = []
    for _, row in test_df.iterrows():
        series = row.iloc[1:].dropna().values.astype(float)
        if len(series) > max_length:
            series = series[:max_length]
        else:
            series = np.pad(series, (0, max_length - len(series)),
                            mode='constant', constant_values=0)
        X_test.append(series)
    X_test = np.array(X_test)

    data_mean = np.nanmean(X_test, axis=0)
    data_std = np.nanstd(X_test, axis=0)
    data_std = np.where(data_std == 0, 1, data_std)
    X_test_normalized = np.where(
        np.isnan(X_test), 0, (X_test - data_mean) / data_std)

    logger.info(f"Loaded M4 Daily test data: X_test shape={X_test.shape}")
    return X_test_normalized


def compute_metrics(original_signal, selected_indices, selected_values):
    ''' 
    Description :Compute evaluation metrics between original and reconstructed signals'''
    x_full = np.arange(len(original_signal))
    sorted_idx = np.argsort(selected_indices)
    sorted_indices = selected_indices[sorted_idx]
    sorted_values = selected_values[sorted_idx]
    if sorted_indices[0] != 0:
        sorted_indices = np.insert(sorted_indices, 0, 0)
        sorted_values = np.insert(sorted_values, 0, original_signal[0])
    if sorted_indices[-1] != len(original_signal) - 1:
        sorted_indices = np.append(sorted_indices, len(original_signal) - 1)
        sorted_values = np.append(sorted_values, original_signal[-1])
    interpolator = interp1d(sorted_indices, sorted_values,
                            kind='linear', fill_value="extrapolate")
    reconstructed_signal = interpolator(x_full)

    mse = mean_squared_error(original_signal, reconstructed_signal)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_signal, reconstructed_signal)
    r2 = r2_score(original_signal, reconstructed_signal)
    correlation, _ = pearsonr(original_signal, reconstructed_signal)
    original_fft = np.abs(np.fft.fft(original_signal))
    reconstructed_fft = np.abs(np.fft.fft(reconstructed_signal))
    spectral_mse = mean_squared_error(original_fft, reconstructed_fft)

    return mse, rmse, mae, r2, correlation, spectral_mse


def evaluate_model(args):
    original_length = 100
    wavelet_name = 'db4'
    dwt_level = 1

    # Loading test data
    X_test_normalized = load_m4_daily(
        args.test_file, max_length=original_length)

    # Loading the trained model
    model_path = 'downsampling_model.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    logger.info("Loading trained model...")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "WaveletDownsamplingModel": WaveletDownsamplingModel,
            "TimeSeriesEmbedding": TimeSeriesEmbedding,
            "DownsampleTransformerBlock": DownsampleTransformerBlock,
            "downsampling_loss": downsampling_loss
        }
    )
    logger.info("Model loaded successfully")

    # Evaluating the model
    logger.info("Evaluating Wavelet-Transformer Model on test dataset")
    wt_model_metrics = {'mse': [], 'rmse': [], 'mae': [], 'r2': [
    ], 'corr': [], 'spectral_mse': [], 'num_points': []}
    len_cA, len_cD = get_wavedec_coeff_lengths(
        original_length, wavelet_name, dwt_level, 'symmetric')
    num_samples = X_test_normalized.shape[0]

    for i in range(num_samples):
        original_signal = X_test_normalized[i]
        _, indices_list = model.call(
            original_signal[np.newaxis, :], training=False, return_indices=True)
        approx_indices = indices_list[0][0].numpy()
        detail_indices = indices_list[-1][0].numpy()
        detail_indices_mapped = (
            detail_indices * (original_length / len_cD)).astype(int)
        selected_indices = np.concatenate(
            [approx_indices, detail_indices_mapped])
        selected_indices = np.clip(
            selected_indices, 0, len(original_signal) - 1)
        selected_indices = np.unique(selected_indices)
        selected_values = original_signal[selected_indices]
        num_selected_points = len(selected_indices)
        wt_model_metrics['num_points'].append(num_selected_points)

        mse, rmse, mae, r2, corr, spectral_mse = compute_metrics(
            original_signal, selected_indices, selected_values)
        wt_model_metrics['mse'].append(mse)
        wt_model_metrics['rmse'].append(rmse)
        wt_model_metrics['mae'].append(mae)
        wt_model_metrics['r2'].append(r2)
        wt_model_metrics['corr'].append(corr)
        wt_model_metrics['spectral_mse'].append(spectral_mse)

        if i % 100 == 0:
            logger.info(f"Processed {i}/{num_samples} samples for Your Model")

    avg_num_points = int(np.mean(wt_model_metrics['num_points']))
    # Rounding the average number of points to the nearest even number
    logger.info(
        f"Average number of downsampled points from Wavelet-Transformer Model: {avg_num_points}")
    if avg_num_points % 2 != 0:
        avg_num_points += 1
    logger.info(
        f"Average number of downsampled points from Wavelet-Transformer Model (adjusted to even): {avg_num_points}")

    # Visualising randomly selected samples
    num_visualize = 3
    visualize_indices = np.random.choice(
        num_samples, num_visualize, replace=False)
    for idx in visualize_indices:
        original_signal = X_test_normalized[idx]
        plt.figure(figsize=(18, 6))

        # Downsampling Algorithm
        _, indices_list = model.call(
            original_signal[np.newaxis, :], training=False, return_indices=True)
        approx_indices = indices_list[0][0].numpy()
        detail_indices = indices_list[-1][0].numpy()
        detail_indices_mapped = (
            detail_indices * (original_length / len_cD)).astype(int)
        selected_indices = np.concatenate(
            [approx_indices, detail_indices_mapped])
        selected_indices = np.clip(
            selected_indices, 0, len(original_signal) - 1)
        selected_indices = np.unique(selected_indices)
        selected_values = original_signal[selected_indices]
        x_full = np.arange(original_length)
        # interpolator = interp1d(selected_indices, selected_values, kind='linear', fill_value="extrapolate")
        interpolator = CubicSpline(selected_indices, selected_values)
        reconstructed_signal = interpolator(x_full)

        plt.subplot(1, 2, 1)
        plt.plot(original_signal, label='Original', alpha=0.7)
        plt.scatter(selected_indices, selected_values,
                    color='red', label='Selected Points')
        plt.plot(reconstructed_signal, label='Reconstructed',
                 alpha=0.7, linestyle='--')
        plt.title(f'Wavelet-Transformer Model - Sample {idx}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'comparison_sample_{idx}.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the downsampling model on the test dataset.")
    parser.add_argument(
        '--test_file', default="M4/Quarterly/Quarterly-train.csv", help="Path to test CSV file")
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    np.random.seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    evaluate_model(args)


if __name__ == "__main__":
    main()

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import os
import argparse
import psutil
import logging
import yaml

from wavelet_transformer_downsampler.wavelet_transformer import WaveletDownsamplingModel, TimeSeriesEmbedding, DownsampleTransformerBlock, get_wavedec_coeff_lengths, downsampling_loss, build_detail_transformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log', mode='w', delay=True)
    ]
)
logger = logging.getLogger(__name__)


def load_model_config(config_path='model_config.yaml'):
    """
    Description:Load model hyperparameters from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Configuration with defaults if file is missing or invalid.
    """
    default_config = {
        'embed_dim': 64,
        'num_heads': 4,
        'ff_dim': 64,
        'num_transformer_blocks': 1,
        'retention_rate': 0.8,
        'global_attention_weight': 0.7,
        'local_attention_weight': 0.3,
        'wavelet_name': 'db4',
        'dwt_level': 1,
        'approx_ds_factor': 2,
        'original_length': 300,
        'normalize_details': True,
        'decomposition_mode': 'symmetric',
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 0.0001
    }
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            logger.warning(
                f"Empty config file at {config_path}, using defaults")
            return default_config
        logger.info(f"Loaded model config from {config_path}: {config}")
        for key, value in default_config.items():
            config.setdefault(key, value)
        return config
    except FileNotFoundError:
        logger.warning(
            f"Model config file {config_path} not found, using defaults")
        return default_config
    except Exception as e:
        logger.error(f"Error loading model config from {config_path}: {e}")
        return default_config


def load_m4_train_datasets(train_files, max_length=300):
    """
    Description:Load M4 training datasets for model training.
    Args:
        train_files (list): List of paths to training CSV files.
        max_length (int): Maximum sequence length for padding/truncating.
    Returns:
        numpy.ndarray: Normalized X_train.
    """
    X_train_all = []

    for train_file in train_files:
        if not os.path.exists(train_file):
            raise FileNotFoundError(
                f"Training dataset file not found: {train_file}")

        # Load training data
        train_df_train = pd.read_csv(train_file)
        train_df = train_df_train
        X_train = []
        for _, row in train_df.iterrows():
            series = row.iloc[1:].dropna().values.astype(float)
            if len(series) > max_length:
                series = series[:max_length]
            else:
                series = np.pad(series, (0, max_length - len(series)),
                                mode='constant', constant_values=0)
            X_train.append(series)
        X_train = np.array(X_train)

        # Normalize training data
        data_mean = np.nanmean(X_train, axis=0)
        data_std = np.nanstd(X_train, axis=0)
        data_std = np.where(data_std == 0, 1, data_std)
        X_train_normalized = np.where(
            np.isnan(X_train), 0, (X_train - data_mean) / data_std)

        X_train_all.append(X_train_normalized)

        logger.info(
            f"Loaded {os.path.basename(train_file)}: X_train shape={X_train.shape}")

    # Concatenate all training datasets
    X_train_combined = np.concatenate(X_train_all, axis=0)

    logger.info(
        f"Combined M4 training data: X_train_combined shape={X_train_combined.shape}")
    return X_train_combined


# Trains the model in batch mode.


def non_stream_pipeline(args):
    config = load_model_config()
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    ff_dim = config['ff_dim']
    num_transformer_blocks = config['num_transformer_blocks']
    retention_rate = config['retention_rate']
    global_attention_weight = config['global_attention_weight']
    local_attention_weight = config['local_attention_weight']
    wavelet_name = config['wavelet_name']
    dwt_level = config['dwt_level']
    approx_ds_factor = config['approx_ds_factor']
    original_length = config['original_length']
    normalize_details = config['normalize_details']
    decomposition_mode = config['decomposition_mode']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']

    # Monitoring memory usage
    logger.info(
        f"Memory usage before model build: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    logger.info("Testing TimeSeriesEmbedding with dummy input...")
    len_cA, signal_coeffs_len = get_wavedec_coeff_lengths(
        original_length, wavelet_name, dwt_level, decomposition_mode)
    dummy_input = tf.zeros((25, signal_coeffs_len, 1))
    embedding_layer = TimeSeriesEmbedding(
        maxlen=signal_coeffs_len, embed_dim=embed_dim)
    dummy_output = embedding_layer(dummy_input)
    logger.info(
        f"Dummy TimeSeriesEmbedding output shape: {dummy_output.shape}")

    # Load only M4 training datasets
    train_files = [args.hourly_train]
    X_train = load_m4_train_datasets(train_files, max_length=original_length)

    # Building the model
    detail_transformer = build_detail_transformer(
        signal_coeffs_len, embed_dim, num_heads, ff_dim, num_transformer_blocks, retention_rate,
        global_attention_weight, local_attention_weight
    )
    model = WaveletDownsamplingModel(
        detail_transformer_model=detail_transformer,
        wavelet_name=wavelet_name,
        approx_ds_factor=approx_ds_factor,
        original_length=original_length,
        signal_coeffs_len=signal_coeffs_len,
        dwt_level=dwt_level,
        normalize_details=normalize_details,
        decomposition_mode=decomposition_mode
    )
    model.build(input_shape=(None, original_length))

    logger.info(
        f"Memory usage after model build: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    # Generate training targets
    y_train = model.call(X_train, training=False, return_indices=False)
    logger.info(f"y_train shape: {y_train.shape}")

    # Compile and train the model
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, clipnorm=0.5)
    model.compile(optimizer=optimizer, loss=downsampling_loss)
    logger.info("Wavelet Downsampling Model Summary:")
    model.summary(line_length=150)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    logger.info("Training Wavelet Downsampling Model")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # Using 20% of training data for validation
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # Saving the models
    logger.info("Saving Models")
    model.save('downsampling_model.keras')
    detail_transformer.save('detail_transformer_model.keras')


def main():
    parser = argparse.ArgumentParser(
        description="Run the downsampling pipeline in stream or non-stream mode.")
    parser.add_argument('--pipeline', choices=['stream', 'non-stream'],
                        default='non-stream', help="Choose pipeline mode: 'stream' or 'non-stream'")
    parser.add_argument('--monthly_train', default="data_test/Monthly-new.csv",
                        help="Path to Monthly training CSV file")
    parser.add_argument('--daily_train', default="data_test/Daily-new.csv",
                        help="Path to Daily training CSV file")
    parser.add_argument('--yearly_train', default="data_test/Yearly-new.csv",
                        help="Path to Yearly training CSV file")
    parser.add_argument('--hourly_train', default="data_test/Hourly-new.csv",
                        help="Path to Hourly training CSV file")
    parser.add_argument('--quarterly_train', default="data_test/Quarterly-new.csv",
                        help="Path to Quarterly training CSV file")
    parser.add_argument('--weekly_train', default="data_test/Weekly-new.csv",
                        help="Path to Weekly training CSV file")
    # parser.add_argument('--daily_test', default="M4/Daily/Daily-test.csv", help="Path to Daily test CSV file")
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    np.random.seed(42)

    # Limiting TensorFlow threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    if args.pipeline == 'non-stream':
        non_stream_pipeline(args)
    else:
        raise NotImplementedError(
            "Stream pipeline is not implemented in this version.")


if __name__ == "__main__":
    main()


import pywt
from keras import layers
import keras
import tensorflow as tf
import numpy as np
import pytest

from .wavelet_transformer import get_wavedec_coeff_lengths, TimeSeriesEmbedding, DownsampleTransformerBlock, WaveletDownsamplingModel, build_detail_transformer, downsampling_loss


# Fixture for common test parameters


@pytest.fixture
def default_params():
    return {
        'signal_length': 100,
        'wavelet_name': 'db4',
        'dwt_level': 1,
        'embed_dim': 64,
        'num_heads': 4,
        'ff_dim': 64,
        'retention_rate': 0.8,
        'global_weight': 0.7,
        'local_weight': 0.3,
        'batch_size': 2,
        'maxlen': 100,
        'approx_ds_factor': 2,
        'normalize_details': True,
        'decomposition_mode': 'symmetric'
    }

# Tests for get_wavedec_coeff_lengths


def test_get_wavedec_coeff_lengths_valid(default_params):
    len_cA, len_cD = get_wavedec_coeff_lengths(
        default_params['signal_length'],
        default_params['wavelet_name'],
        default_params['dwt_level'],
        default_params['decomposition_mode']
    )
    assert isinstance(len_cA, int)
    assert isinstance(len_cD, int)
    assert len_cA > 0
    assert len_cD > 0
    assert len_cA == len_cD  # For db4 wavelet at level 1, lengths should be equal


def test_get_wavedec_coeff_lengths_invalid_level(default_params):
    with pytest.raises(ValueError, match="Decomposition level must be >= 0"):
        get_wavedec_coeff_lengths(
            default_params['signal_length'],
            default_params['wavelet_name'],
            -1,
            default_params['decomposition_mode']
        )


def test_get_wavedec_coeff_lengths_wavelet_object(default_params):
    wavelet = pywt.Wavelet(default_params['wavelet_name'])
    len_cA, len_cD = get_wavedec_coeff_lengths(
        default_params['signal_length'],
        wavelet,
        default_params['dwt_level'],
        default_params['decomposition_mode']
    )
    assert isinstance(len_cA, int)
    assert isinstance(len_cD, int)

# Tests for TimeSeriesEmbedding


def test_time_series_embedding_shape(default_params):
    embedding = TimeSeriesEmbedding(
        maxlen=default_params['maxlen'], embed_dim=default_params['embed_dim'])
    input_data = tf.zeros(
        (default_params['batch_size'], default_params['maxlen'], 1))
    output = embedding(input_data)
    assert output.shape == (
        default_params['batch_size'], default_params['maxlen'], default_params['embed_dim'])


def test_time_series_embedding_invalid_input_shape(default_params):
    embedding = TimeSeriesEmbedding(
        maxlen=default_params['maxlen'], embed_dim=default_params['embed_dim'])
    invalid_input = tf.zeros(
        (default_params['batch_size'], default_params['maxlen'] + 10, 1))
    with pytest.raises(ValueError, match=f"Expected input shape \\(batch_size, {default_params['maxlen']}, 1\\)"):
        embedding.build(invalid_input.shape)


def test_time_series_embedding_pos_encoding(default_params):
    embedding = TimeSeriesEmbedding(
        maxlen=default_params['maxlen'], embed_dim=default_params['embed_dim'])
    pos_encoding = embedding.get_sinusoidal_pos_encoding(
        default_params['maxlen'], default_params['embed_dim'])
    assert pos_encoding.shape == (
        1, default_params['maxlen'], default_params['embed_dim'])
    assert not tf.reduce_all(pos_encoding == 0)  # Ensure non-zero encoding

# Tests for DownsampleTransformerBlock


def test_downsample_transformer_block_shape(default_params):
    transformer = DownsampleTransformerBlock(
        embed_dim=default_params['embed_dim'],
        num_heads=default_params['num_heads'],
        ff_dim=default_params['ff_dim'],
        retention_rate=default_params['retention_rate'],
        global_weight=default_params['global_weight'],
        local_weight=default_params['local_weight']
    )
    input_data = tf.random.normal(
        (default_params['batch_size'], default_params['maxlen'], default_params['embed_dim']))
    transformer.build(input_data.shape)
    output, indices = transformer(input_data)
    expected_seq_len = max(
        1, int(round(default_params['maxlen'] * default_params['retention_rate'])))
    assert output.shape == (
        default_params['batch_size'], expected_seq_len, default_params['embed_dim'])
    assert indices.shape == (default_params['batch_size'], expected_seq_len)


def test_downsample_transformer_block_invalid_retention_rate(default_params):
    with pytest.raises(ValueError, match="retention_rate must be between 0 and 1"):
        DownsampleTransformerBlock(
            embed_dim=default_params['embed_dim'],
            num_heads=default_params['num_heads'],
            ff_dim=default_params['ff_dim'],
            retention_rate=1.5
        )


def test_downsample_transformer_block_weight_sum(default_params):
    with pytest.raises(ValueError, match="global_weight.*and local_weight.*must sum to 1.0"):
        DownsampleTransformerBlock(
            embed_dim=default_params['embed_dim'],
            num_heads=default_params['num_heads'],
            ff_dim=default_params['ff_dim'],
            retention_rate=default_params['retention_rate'],
            global_weight=0.8,
            local_weight=0.3
        )


def test_downsample_transformer_block_invalid_input_shape(default_params):
    transformer = DownsampleTransformerBlock(
        embed_dim=default_params['embed_dim'],
        num_heads=default_params['num_heads'],
        ff_dim=default_params['ff_dim'],
        retention_rate=default_params['retention_rate']
    )
    invalid_input = tf.zeros(
        (default_params['batch_size'], default_params['maxlen']))
    with pytest.raises(ValueError, match="Expected 3D input shape"):
        transformer.build(invalid_input.shape)

# Tests for WaveletDownsamplingModel


def test_wavelet_downsampling_model(default_params):
    len_cA, len_cD = get_wavedec_coeff_lengths(
        default_params['signal_length'],
        default_params['wavelet_name'],
        default_params['dwt_level'],
        default_params['decomposition_mode']
    )
    detail_transformer = build_detail_transformer(
        input_seq_len=len_cD,
        embed_dim=default_params['embed_dim'],
        num_heads=default_params['num_heads'],
        ff_dim=default_params['ff_dim'],
        num_transformer_blocks=1,
        retention_rate=default_params['retention_rate'],
        global_weight=default_params['global_weight'],
        local_weight=default_params['local_weight']
    )
    model = WaveletDownsamplingModel(
        detail_transformer_model=detail_transformer,
        wavelet_name=default_params['wavelet_name'],
        approx_ds_factor=default_params['approx_ds_factor'],
        original_length=default_params['signal_length'],
        signal_coeffs_len=len_cD,
        dwt_level=default_params['dwt_level'],
        normalize_details=default_params['normalize_details'],
        decomposition_mode=default_params['decomposition_mode']
    )
    input_data = tf.random.normal(
        (default_params['batch_size'], default_params['signal_length'], 1))
    model.build(input_data.shape)
    output = model(input_data, training=False)
    expected_output_len = max(
        1, (len_cA - default_params['approx_ds_factor']) // default_params['approx_ds_factor'] + 1)
    expected_output_len += detail_transformer.compute_output_shape((None, len_cD, 1))[
        0][-1]
    assert output.shape == (default_params['batch_size'], expected_output_len)


def test_wavelet_downsampling_model_with_indices(default_params):
    len_cA, len_cD = get_wavedec_coeff_lengths(
        default_params['signal_length'],
        default_params['wavelet_name'],
        default_params['dwt_level'],
        default_params['decomposition_mode']
    )
    detail_transformer = build_detail_transformer(
        input_seq_len=len_cD,
        embed_dim=default_params['embed_dim'],
        num_heads=default_params['num_heads'],
        ff_dim=default_params['ff_dim'],
        num_transformer_blocks=1,
        retention_rate=default_params['retention_rate'],
        global_weight=default_params['global_weight'],
        local_weight=default_params['local_weight']
    )
    model = WaveletDownsamplingModel(
        detail_transformer_model=detail_transformer,
        wavelet_name=default_params['wavelet_name'],
        approx_ds_factor=default_params['approx_ds_factor'],
        original_length=default_params['signal_length'],
        signal_coeffs_len=len_cD,
        dwt_level=default_params['dwt_level'],
        normalize_details=default_params['normalize_details'],
        decomposition_mode=default_params['decomposition_mode']
    )
    input_data = tf.random.normal(
        (default_params['batch_size'], default_params['signal_length'], 1))
    model.build(input_data.shape)
    output, indices_list = model(
        input_data, training=False, return_indices=True)
    assert len(indices_list) >= 1  # At least one set of indices (approx)
    assert all(isinstance(indices, tf.Tensor) for indices in indices_list)
    assert output.shape[0] == default_params['batch_size']

# Tests for downsampling_loss


def downsampling_loss(y_true, y_pred):
    ''' Description: Downsampling loss function that combines MSE and frequency domain loss,
        the frequency-domain loss ensures preservation of spectral characteristics,
        which is valuable for time series '''
    mse_loss = tf.reduce_mean(keras.losses.mean_squared_error(
        y_true, y_pred))  # Reduce across batch
    y_true_fft = tf.abs(tf.signal.fft(tf.cast(y_true, tf.complex64)))
    y_pred_fft = tf.abs(tf.signal.fft(tf.cast(y_pred, tf.complex64)))
    freq_loss = tf.reduce_mean(tf.square(y_true_fft - y_pred_fft))
    return mse_loss + 0.5 * freq_loss

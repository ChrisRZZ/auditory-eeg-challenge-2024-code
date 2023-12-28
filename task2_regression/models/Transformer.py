

import tensorflow as tf

from task2_regression.models.vlaai import pearson_tf, pearson_tf_non_averaged

@tf.function
def pearson_loss_cut(y_true, y_pred, axis=1):
    """Pearson loss function.

    Parameters
    ----------
    y_true: tf.Tensor
        True values. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted values. Shape is (batch_size, time_steps, n_features)

    Returns
    -------
    tf.Tensor
        Pearson loss.
        Shape is (batch_size, 1, n_features)
    """
    return -pearson_tf(y_true[:, : tf.shape(y_pred)[1], :], y_pred, axis=axis)


@tf.function
def pearson_metric_cut(y_true, y_pred, axis=1):
    """Pearson metric function.

    Parameters
    ----------
    y_true: tf.Tensor
        True values. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted values. Shape is (batch_size, time_steps, n_features)

    Returns
    -------
    tf.Tensor
        Pearson metric.
        Shape is (batch_size, 1, n_features)
    """
    return pearson_tf(y_true[:, : tf.shape(y_pred)[1], :], y_pred, axis=axis)

@tf.function
def pearson_metric_cut_non_averaged(y_true, y_pred, axis=1):
    """Pearson metric function.

    Parameters
    ----------
    y_true: tf.Tensor
        True values. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted values. Shape is (batch_size, time_steps, n_features)

    Returns
    -------
    tf.Tensor
        Pearson metric.
        Shape is (batch_size, 1, n_features)
    """
    return pearson_tf_non_averaged(y_true[:, : tf.shape(y_pred)[1], :], y_pred, axis=axis)


def build_transformer_model(input_shape, num_heads, num_transformer_blocks, dff, dropout_rate):
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    for _ in range(num_transformer_blocks):
        # Multi-head attention with residual connection and layer normalization
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
        attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
        attention_output = tf.keras.layers.Add()([x, attention_output])  # Residual connection
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output)

        # Feed-forward network with residual connection and layer normalization
        ffn_output = tf.keras.layers.Dense(dff, activation='relu')(x)
        ffn_output = tf.keras.layers.Dense(input_shape[-1])(ffn_output)
        ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
        ffn_output = tf.keras.layers.Add()([x, ffn_output])  # Residual connection
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output)

    outputs = tf.keras.layers.Dense(input_shape[-1])(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def transformer_model(input_shape, num_heads, num_transformer_blocks, dff, dropout_rate):
    model = build_transformer_model(input_shape, num_heads, num_transformer_blocks, dff, dropout_rate)
    model.compile(
        tf.keras.optimizers.legacy.Adam(),  # Use legacy Adam optimizer
        loss=pearson_loss_cut,
        metrics=[pearson_metric_cut]
    )
    return model




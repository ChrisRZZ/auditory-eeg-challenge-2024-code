""" This module contains linear backward model"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Conv1D, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from task2_regression.models.vlaai import pearson_tf, pearson_tf_non_averaged
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, LSTM, GRU, Dense, Concatenate
from tensorflow.keras.models import Model

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



def simple_linear_model(integration_window=32, nb_filters=1, nb_channels=64):
    inp = tf.keras.layers.Input(
        (
            None,
            nb_channels,
        )
    )
    out = tf.keras.layers.Conv1D(nb_filters, integration_window)(inp)
    model = tf.keras.models.Model(inputs=[inp], outputs=[out])
    model.compile(
        tf.keras.optimizers.Adam(),
        loss=pearson_loss_cut,
        metrics=[pearson_metric_cut]
    )
    return model

def improved_model(integration_window=32, nb_filters=1, nb_channels=64, output_channels=10):
    inp = tf.keras.layers.Input((None, nb_channels))
    
    x = tf.keras.layers.Conv1D(nb_filters, integration_window)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Example of adding an extra convolutional layer
    x = tf.keras.layers.Conv1D(nb_filters * 2, integration_window)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Dense layer
    x = tf.keras.layers.Dense(64)(x)
    
    out = tf.keras.layers.Dense(nb_channels)(x)  # Adjust the output layer as per your requirement

    model = tf.keras.models.Model(inputs=[inp], outputs=[out])
    
    # Reshape the output to match the expected target shape [None, None, 10]
    x = tf.keras.layers.Dense(output_channels, activation=None)(x)
    x = tf.keras.layers.Reshape((-1, output_channels))(x)

    model = tf.keras.models.Model(inputs=[inp], outputs=[x])
    
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=pearson_loss_cut,
        metrics=[pearson_metric_cut]
    )
    return model

def improved_LSTM_model(integration_window=32, nb_filters=1, nb_channels=64, output_channels=10):
    inp = tf.keras.layers.Input((None, nb_channels))
    
    x = tf.keras.layers.Conv1D(nb_filters, integration_window, activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Example of adding an extra convolutional layer
    x = tf.keras.layers.Conv1D(nb_filters * 2, integration_window, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # LSTM Layer - Captures temporal dependencies
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Dense layer
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    out = tf.keras.layers.Dense(nb_channels)(x)  # Adjust the output layer as per your requirement

    model = tf.keras.models.Model(inputs=[inp], outputs=[out])
    
    # Reshape the output to match the expected target shape [None, None, 10]
    x = tf.keras.layers.Dense(output_channels, activation=None)(x)
    x = tf.keras.layers.Reshape((-1, output_channels))(x)

    model = tf.keras.models.Model(inputs=[inp], outputs=[x])
    
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=pearson_loss_cut,
        metrics=[pearson_metric_cut]
    )
    return model

def improved_GRU_model(integration_window=32, nb_filters=1, nb_channels=64, output_channels=10):
    inp = tf.keras.layers.Input((None, nb_channels))
    
    # Convolutional layers
    x = tf.keras.layers.Conv1D(nb_filters, integration_window, activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Conv1D(nb_filters * 2, integration_window, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # GRU Layer - Captures temporal dependencies
    x = tf.keras.layers.GRU(64, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Dense layer
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    out = tf.keras.layers.Dense(nb_channels)(x)  # Adjust the output layer as per your requirement

    # Reshape the output to match the expected target shape [None, None, 10]
    x = tf.keras.layers.Dense(output_channels, activation=None)(x)
    x = tf.keras.layers.Reshape((-1, output_channels))(x)

    model = tf.keras.models.Model(inputs=[inp], outputs=[x])
    
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=pearson_loss_cut,
        metrics=[pearson_metric_cut]
    )
    return model

def simple_linear_model_stimulus(integration_window=32, nb_filters=1, nb_channels=64):
    inp = tf.keras.layers.Input(
        (
            None,
            nb_channels,
        )


    )
    # env = abs(s)
    # f0= np.phase(s)
    # f0 = np.angle(s)

    # reconstruct env
    # reconsturct f0
    # reconstructed s = real(reconstructed_env .*exp(1j*reconstructed_f0))./ np.max(abs(reconstructed_env))

    out = tf.keras.layers.Conv1D(nb_filters, integration_window)(inp)
    model = tf.keras.models.Model(inputs=[inp], outputs=[out])
    model.compile(
        tf.keras.optimizers.Adam(),
        loss=pearson_loss_cut,
        metrics=[pearson_metric_cut]
    )
    return model

def combined_LSTM_GRU_model(integration_window=32, nb_filters=1, nb_channels=64, output_channels=10):
    inp = Input(shape=(None, nb_channels))
    
    # Shared convolutional layers
    x = Conv1D(nb_filters, integration_window, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Conv1D(nb_filters * 2, integration_window, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # LSTM branch
    lstm_out = LSTM(64, return_sequences=True)(x)
    lstm_out = Dropout(0.5)(lstm_out)

    # GRU branch
    gru_out = GRU(64, return_sequences=True)(x)
    gru_out = Dropout(0.5)(gru_out)

    # Combine the outputs of LSTM and GRU branches
    combined = Concatenate(axis=-1)([lstm_out, gru_out])

    # Dense layers after combining
    combined = Dense(64, activation='relu')(combined)
    combined = Dense(output_channels)(combined)

    # Reshape the output to match the expected target shape [None, None, 10]
    combined = Dense(output_channels, activation=None)(combined)
    combined = tf.keras.layers.Reshape((-1, output_channels))(combined)

    # Create and compile the combined model
    model = Model(inputs=inp, outputs=combined)
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=pearson_loss_cut,
        metrics=[pearson_metric_cut]
    )

    return model



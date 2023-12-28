# Import necessary libraries
import tensorflow as tf
import os
import json
import glob
import logging
import numpy as np
# set gpu private
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf

import sys
# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import scipy.stats
import numpy as np

from task2_regression.models.Transformer import build_transformer_model, transformer_model
from util.dataset_generator import DataGenerator, create_tf_dataset
from task2_regression.models.linear import simple_linear_model, pearson_loss_cut, pearson_metric_cut, pearson_metric_cut_non_averaged
from util.dataset_generator import DataGenerator, create_tf_dataset

def evaluate_model(model, test_dict):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        # evaluate model
        ds = [x for x in ds_test]
        eeg = tf.concat([x[0] for x in ds], axis=0)
        labels = tf.concat([x[1] for x in ds], axis=0)

        reconstructions = model.predict(eeg)
        correlations = np.squeeze(pearson_metric_cut_non_averaged(labels, reconstructions))

        # calculate pearson correlation per band
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
        evaluation[subject]["pearson_correlation_per_band"] = np.mean(correlations, axis=0).tolist()

    return evaluation

if __name__ == "__main__":
    # Set parameters
    fs = 64
    window_length = 60 * fs  # 10 seconds
    hop_length = 30 * fs
    epochs = 100
    patience = 5
    batch_size = 64
    only_evaluate = False

    # Get the path to the config file
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    util_folder = os.path.join(os.path.dirname(task_folder), "util")
    config_path = os.path.join(util_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    data_folder = os.path.join(config["dataset_folder"], config["derivatives_folder"], config["split_folder"])
    stimulus_features = ["mel"]
    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_transformer")
    os.makedirs(results_folder, exist_ok=True)

    # Create a dataset generator for each training subject
    all_subs = list(set([os.path.basename(x).split("_-_")[1] for x in glob.glob(os.path.join(data_folder, "train_-_*"))]))

    # Create a transformer model
    model = transformer_model(input_shape=(window_length, 64), num_heads=4, num_transformer_blocks=2, dff=128, dropout_rate=0.1)
    model.summary()
    model_path = os.path.join(results_folder, f"model.h5")
    training_log_filename = f"training_log.csv"
    results_filename = f'eval.json'

    if only_evaluate:
        # Load weights
        model.load_weights(model_path)
    else:
        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        train_generator = DataGenerator(train_files, window_length)
        dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        val_generator = DataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

        # Train the model
        model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
                tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            ],
            workers=tf.data.AUTOTUNE,
            use_multiprocessing=True
        )

    # Evaluate the model on the test set
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
    datasets_test = {}
    for sub in subjects:
        files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
        test_generator = DataGenerator(files_test_sub, window_length)
        datasets_test[sub] = create_tf_dataset(test_generator, window_length, None, hop_length, batch_size=1, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

    evaluation = evaluate_model(model, datasets_test)

    # Save the results in a json encoded file
    results_path = os.path.join(results_folder, results_filename)
    with open(results_path, "w") as fp:
        json.dump(evaluation, fp)
    logging.info(f"Results saved at {results_path}")

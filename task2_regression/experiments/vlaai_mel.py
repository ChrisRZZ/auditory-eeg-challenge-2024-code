"""Example experiment for the VLAAI model."""
import glob
import json
import logging
import os
# set gpu private
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
import tensorflow as tf


import numpy as np
import sys
# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from task2_regression.models.vlaai import vlaai, pearson_loss, pearson_metric, pearson_tf_non_averaged
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
        correlations = np.squeeze(pearson_tf_non_averaged(labels, reconstructions))

        # calculate pearson correlation per band

        results = model.evaluate(ds_test, verbose=2)

        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))

        evaluation[subject]["pearson_correlation_per_band"] = np.mean(correlations, axis=0).tolist()
        # metrics = model.metrics_names
        # evaluation[subject] = dict(zip(metrics, results))
    return evaluation



if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    fs= 64
    window_length = 5 * fs  # 10 seconds
    # Hop length between two consecutive decision windows
    hop_length = 1*fs
    epochs = 100
    patience = 10
    batch_size = 10
    only_evaluate = False # If it is true, will only evaluate
    training_log_filename = "training_log.csv"
    results_filename = 'eval.json'


   # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    util_folder = os.path.join(os.path.dirname(task_folder), "util")
    config_path = os.path.join(util_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test

    data_folder = os.path.join(config["dataset_folder"],config["derivatives_folder"],  config["split_folder"])
    stimulus_features = ["mel"]
    features = ["eeg"] + stimulus_features
    

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "results_vlaai_mel")
    os.makedirs(results_folder, exist_ok=True)

    # create the model
    model = vlaai()
    #model.compile(tf.keras.optimizers.Adam(), loss=pearson_loss, metrics=[pearson_metric])
    model.compile(tf.keras.optimizers.legacy.Adam(), loss=pearson_loss, metrics=[pearson_metric]) #this is for M1/M2 Mac
    model_path = os.path.join(results_folder, "model.h5")
    #model_path = os.path.join(results_folder, "vlaai.h5")
    
    if only_evaluate:

        model.load_weights(model_path)
    else:
        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # Create list of numpy array files
        train_generator = DataGenerator(train_files, window_length)
        dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features ]
        val_generator = DataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))
        
        # Adjust the number of epochs
        epochs = 50  # Example: Reduce the number of epochs if it's currently very high

        # Further reduce the patience in EarlyStopping
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=3,  # Further reduce patience
            restore_best_weights=True,
            monitor='val_loss',  # Change to 'val_loss' if more appropriate
            verbose=1  # Enable verbose output
        )

        # Optional: Add a learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1
        )

        # Update the model.fit call with new parameters
        model.fit(
            dataset_train,
            epochs=epochs,  # Keep the epochs reasonable
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
                early_stopping_callback,  # Updated EarlyStopping callback
                lr_scheduler  # Optional learning rate scheduler
            ],
            workers=tf.data.AUTOTUNE,
            use_multiprocessing=True
        )

    # Evaluate the model on test set
    # Create a dataset generator for each test subject
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    # Get all different subjects from the test set
    subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
    datasets_test = {}
    # Create a generator for each subject
    for sub in subjects:
        files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
        test_generator = DataGenerator(files_test_sub, window_length)
        datasets_test[sub] = create_tf_dataset(test_generator, window_length, None, hop_length,    batch_size=64, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

    evaluation = evaluate_model(model, datasets_test)

    # We can save our results in a json encoded file
    results_path = os.path.join(results_folder, results_filename)
    with open(results_path, "w") as fp:
        json.dump(evaluation, fp)
    logging.info(f"Results saved at {results_path}")

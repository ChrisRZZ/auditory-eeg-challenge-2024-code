"""Example experiment for a linear baseline method."""
import glob
import json
import logging
import os
# set gpu private
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf

import sys
# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import scipy.stats
import numpy as np


from task2_regression.models.linear import simple_linear_model, pearson_loss_cut, pearson_metric_cut, pearson_metric_cut_non_averaged
from util.dataset_generator import DataGenerator, create_tf_dataset


# def evaluate_model(model, test_dict):
#     """Evaluate a model.

#     Parameters
#     ----------
#     model: tf.keras.Model
#         Model to evaluate.
#     test_dict: dict
#         Mapping between a subject and a tf.data.Dataset containing the test
#         set for the subject.

#     Returns
#     -------
#     dict
#         Mapping between a subject and the loss/evaluation score on the test set
#     """
#     evaluation = {}
#     for subject, ds_test in test_dict.items():
#         logging.info(f"Scores for subject {subject}:")
#            # evaluate model
#         ds = [x for x in ds_test]
#         eeg = tf.concat([ x[0] for x in ds], axis=0)
#         labels =tf.concat([ x[1] for x in ds], axis=0)


#         reconstructions = model.predict(eeg)
#         correlations = np.squeeze(pearson_metric_cut_non_averaged(labels, reconstructions))

#         # calculate pearson correlation per band

#         results = model.evaluate(ds_test, verbose=2)

#         metrics = model.metrics_names
#         evaluation[subject] = dict(zip(metrics, results))


#         evaluation[subject]["pearson_correlation_per_band"] = np.mean(correlations, axis=0).tolist()
#         # metrics = model.metrics_names
#         # evaluation[subject] = dict(zip(metrics, results))
#     return evaluation

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_dict):
    # Create subfolders
    correlation_folder = os.path.join(results_folder, "correlations")
    reconstruction_folder = os.path.join(results_folder, "reconstructions")
    pearson_correlation_folder = os.path.join(results_folder, "Pearson_correlation")  # New folder for Pearson correlation plots
    os.makedirs(correlation_folder, exist_ok=True)
    os.makedirs(reconstruction_folder, exist_ok=True)
    os.makedirs(pearson_correlation_folder, exist_ok=True)  # Ensure the new folder is created
    
    # Dictionaries to hold the metrics for each subject
    mse_scores = {}
    mae_scores = {}
    rmse_scores = {}
    
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        ds = [x for x in ds_test]
        eeg = tf.concat([x[0] for x in ds], axis=0)
        labels = tf.concat([x[1] for x in ds], axis=0)

        reconstructions = model.predict(eeg)
        correlations = np.squeeze(pearson_metric_cut_non_averaged(labels, reconstructions))

        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))

        pearson_correlation_per_band = np.mean(correlations, axis=0)
        evaluation[subject]["pearson_correlation_per_band"] = pearson_correlation_per_band.tolist()

        # Check shapes and print them for debugging
        print(f"Shapes before flatten - labels: {labels.shape}, reconstructions: {reconstructions.shape}")

        # Handling potential shape mismatch
        # Trim the longer tensor if there is a mismatch
        min_length = min(tf.size(labels), tf.size(reconstructions))
        labels = tf.reshape(labels, [-1])[:min_length]
        reconstructions = tf.reshape(reconstructions, [-1])[:min_length]

        # Convert tensors to numpy arrays for sklearn metrics
        labels_flat = labels.numpy()
        reconstructions_flat = reconstructions.numpy()

        # Calculate additional metrics and convert them to Python floats
        mse = float(mean_squared_error(labels_flat, reconstructions_flat))
        mae = float(mean_absolute_error(labels_flat, reconstructions_flat))
        rmse = float(np.sqrt(mse))  # Root Mean Squared Error

        evaluation[subject]["mse"] = mse
        evaluation[subject]["mae"] = mae
        evaluation[subject]["rmse"] = rmse
        
        # Store the metrics in the dictionaries using the subject as the key
        mse_scores[subject] = mse
        mae_scores[subject] = mae
        rmse_scores[subject] = rmse
        
        # Plotting Pearson Correlation per Band
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(pearson_correlation_per_band)), pearson_correlation_per_band)
        plt.title(f'Pearson Correlation per Band for Subject {subject}')
        plt.xlabel('Band')
        plt.ylabel('Pearson Correlation')
        plt.savefig(os.path.join(pearson_correlation_folder, f'pearson_correlation_{subject}.png'))  # Save in the new folder
        plt.close()
        
        # Correlation Heatmaps
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.mean(correlations, axis=0).reshape(-1, 1), annot=True, cmap='coolwarm')
        plt.title(f'Heatmap of Pearson Correlation for Subject {subject}')
        plt.ylabel('Band')
        plt.xlabel('Correlation')
        plt.savefig(os.path.join(correlation_folder, f'heatmap_{subject}.png'))
        plt.close()
        
        # Example reconstructions
        for i in range(3):  # Plot first 3 examples
            plt.figure(figsize=(12, 4))
            plt.plot(eeg[i], label='Original')
            plt.plot(reconstructions[i], label='Reconstructed')
            plt.title(f'EEG Reconstruction for Subject {subject} - Example {i+1}')
            plt.legend()
            plt.savefig(os.path.join(reconstruction_folder, f'reconstruction_{subject}_{i+1}.png'))
            plt.close()
            
        #  # Visualization of predictions
        # for i in range(min(3, len(eeg))):  # Plot up to 3 examples
        #     plt.figure(figsize=(12, 4))
        #     plt.plot(eeg[i], label='Actual EEG')
        #     plt.plot(reconstructions[i], label='Predicted EEG')
        #     plt.title(f'EEG Signal and Reconstruction for Subject {subject} - Example {i+1}')
        #     plt.legend()
        #     plt.savefig(os.path.join(reconstruction_folder, f'prediction_{subject}_{i+1}.png'))
        #     plt.close()
        
        # Diagnostic print statements to verify the shapes at runtime
        print(f"Shape of eeg: {eeg.shape}")
        print(f"Shape of reconstructions: {reconstructions.shape}")

        # Visualization of predictions
        for i in range(min(3, len(eeg))):  # Plot up to 3 examples
            plt.figure(figsize=(12, 4))
            
            # Check if the reconstructions tensor has the expected number of dimensions
            if len(reconstructions.shape) == 3 and reconstructions.shape[1] == eeg.shape[1]:
                # Plot predicted EEG for the first channel
                plt.plot(reconstructions[i, :, 0], color='k', linestyle='--', label='Predicted EEG (Channel 1)')

            # Plot actual EEG signals for all channels in the first feature dimension
            for ch in range(eeg.shape[2]):  # Iterate over the third dimension (channels)
                plt.plot(eeg[i, :, ch], label=f'Channel {ch+1}' if ch in [0, eeg.shape[2]//2, eeg.shape[2]-1] else None)
            
            plt.title(f'EEG Signal and Reconstruction for Subject {subject} - Example {i+1}')
            plt.xlabel('Time Points')
            plt.ylabel('EEG Signal')
            plt.legend()
            plt.savefig(os.path.join(reconstruction_folder, f'prediction_{subject}_{i+1}.png'))
            plt.close()
            
    # Return the evaluation dictionary and the metrics
    return evaluation, mse_scores, mae_scores, rmse_scores

if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    fs = 64
    window_length = 60 * fs  # 10 seconds
    # Hop length between two consecutive decision windows
    hop_length = 30*fs
    epochs = 100
    patience = 5
    batch_size = 64
    only_evaluate = False # if true, only evaluate the model

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
    results_folder = os.path.join(experiments_folder, "results_linear_baseline")
    os.makedirs(results_folder, exist_ok=True)

    # train a sub dependent model for each sub
    # Create a dataset generator for each training subject
    # Get all different subjects from the training set
    all_subs = list(
        set([os.path.basename(x).split("_-_")[1] for x in glob.glob(os.path.join(data_folder, "train_-_*"))]))


    # create a simple linear model
    model = simple_linear_model(integration_window = int(fs*0.25), nb_filters=10)
    model.summary()
    model_path = os.path.join(results_folder, f"model.h5")
    training_log_filename = f"training_log.csv"
    results_filename = f'eval.json'


    if only_evaluate:
        # load weights
        model.load_weights(model_path)
    else:

        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features ]
        # Create list of numpy array files
        train_generator = DataGenerator(train_files, window_length)
        dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        val_generator = DataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

        # # Train the model
        # model.fit(
        #     dataset_train,
        #     epochs=epochs,
        #     validation_data=dataset_val,
        #     callbacks=[
        #         tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        #         tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
        #         tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
        #     ],
        #     workers = tf.data.AUTOTUNE,
        #     use_multiprocessing=True

        # )
        # Train the model
        history = model.fit(
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
        
        # Plot training history for loss
        plt.figure(figsize=(12, 5))

        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        # Plot Pearson Correlation Over Epochs
        plt.subplot(1, 2, 2)
        plt.plot(history.history['pearson_metric_cut'], label='Training Pearson Correlation')
        plt.plot(history.history['val_pearson_metric_cut'], label='Validation Pearson Correlation')
        plt.title('Pearson Correlation Over Epochs')
        plt.ylabel('Pearson Correlation')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, 'training_history.png'))
        plt.close()

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
        datasets_test[sub] = create_tf_dataset(test_generator, window_length, None, hop_length, batch_size=1, data_types=(tf.float32, tf.float32), feature_dims=(64, 10))

    # Evaluate the model
    #evaluation = evaluate_model(model, datasets_test)
    
    # After model evaluation
    evaluation, mse_scores, mae_scores, rmse_scores = evaluate_model(model, datasets_test)

    # Convert the metric dictionaries to sorted lists for plotting
    subjects_sorted = sorted(mse_scores.keys(), key=lambda x: int(x.split('-')[1]))
    mse_sorted = [mse_scores[sub] for sub in subjects_sorted]
    mae_sorted = [mae_scores[sub] for sub in subjects_sorted]
    rmse_sorted = [rmse_scores[sub] for sub in subjects_sorted]
    
    # Plotting the metrics
    plt.figure(figsize=(20, 6))  # Adjusted figure size for better visibility

    # Function to decide which x-ticks to show
    def select_x_ticks(x_ticks, step=5):
        return [tick if idx % step == 0 else '' for idx, tick in enumerate(x_ticks)]

    # Plot MSE
    plt.subplot(1, 3, 1)
    plt.bar(subjects_sorted, mse_sorted)
    plt.title('MSE by Subject')
    plt.xlabel('Subject')
    plt.xticks(range(len(subjects_sorted)), select_x_ticks(subjects_sorted), rotation=90)  # Show only selected x-ticks
    plt.ylabel('MSE')

    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.bar(subjects_sorted, mae_sorted)
    plt.title('MAE by Subject')
    plt.xlabel('Subject')
    plt.xticks(range(len(subjects_sorted)), select_x_ticks(subjects_sorted), rotation=90)
    plt.ylabel('MAE')

    # Plot RMSE
    plt.subplot(1, 3, 3)
    plt.bar(subjects_sorted, rmse_sorted)
    plt.title('RMSE by Subject')
    plt.xlabel('Subject')
    plt.xticks(range(len(subjects_sorted)), select_x_ticks(subjects_sorted), rotation=90)
    plt.ylabel('RMSE')

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'metrics_by_subject.png'))
    plt.close()

    # We can save our results in a json encoded file
    results_path = os.path.join(results_folder, results_filename)
    # with open(results_path, "w") as fp:
    #     json.dump(evaluation, fp)
    # logging.info(f"Results saved at {results_path}")
    
    # When saving the evaluation results
with open(results_path, "w") as fp:
    json.dump(evaluation, fp)


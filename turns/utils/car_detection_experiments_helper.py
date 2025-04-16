import os
import numpy as np
import pandas as pd

def evaluate_and_save_dbscan_results(dbscan_model, prediction_location, err, yaw, car1_velocity):
    path = 'car_mapping_experiments/dbscan_results/'
    file_name = 'dbscan_results.csv'
    eps = dbscan_model.eps
    min_samples = dbscan_model.min_samples
    unique_labels = np.unique(dbscan_model.labels_)
    num_clusters = len(unique_labels) - 1  # because label == -1 is for noise
    if prediction_location is not None:
        prediction_location = [round(prediction_location[0], 3), round(prediction_location[1], 3)]
        stats = {'Eps': eps, 'Min Samples': min_samples, 'Number Of Clusters Found': num_clusters,
                 'Error': round(err, 3), 'Yaw': yaw, 'Car 1 Velocity':round(car1_velocity, 3),
                 'Predicted Location': prediction_location, 'Car 2 Velocity': 0.0}

        save_results_to_csv(path + file_name, stats)
#
#
# endregion

#
def save_results_to_csv(path, results: dict):
    if not os.path.isfile(path):
        df = pd.DataFrame([results])
        df.to_csv(path, index=False)
    else:
        existing_data = pd.read_csv(path)
        result_df = pd.DataFrame([results])
        updated_data = pd.concat([existing_data, result_df], ignore_index=True)
        updated_data.to_csv(path, index=False)

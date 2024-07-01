# this script uploads the dataset and convert 4D channel matrices into 2D vectors
!pip install argparse
import os
import argparse
import numpy as np
import pickle
from scipy import io
import time 
#loading matlab files
def get_features(path, features_dim):
    data = io.loadmat(path)
    return data['datat'][:, :features_dim]

def get_labels(path, features_dim, labels_dim):
    data = io.loadmat(path)
    return data['Labels']

def set_dims(dataset_path):
    with open(os.path.join(dataset_path, "count.txt"), "r") as f:
        return [int(i) for i in f.read().split("\n") if i != ""]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="D1", help="Name of the dataset")
    args, _ = parser.parse_known_args()
    dataset = args.dataset
    
    dataset_path = "C:/path/To/your/data/{}/".format(dataset)

    # Passing the correct dataset path to set_dims function
    features_dim, labels_dim = set_dims(dataset_path)

    train_features = get_features(os.path.join(dataset_path, "x_train.mat"), features_dim)
    train_features=np.transpose( train_features,(3,0,1,2))
    #flatten the channel matrix to be 2D vector
    train_features=train_features.reshape(6000, -1)
    data_type = type(train_features)
    train_labels = get_labels(os.path.join(dataset_path, "y_train.mat"), features_dim, labels_dim)
    with open(os.path.join(dataset_path, "{}-train-features.pkl".format(dataset)), "wb") as f:
        pickle.dump(train_features, f)
    with open(os.path.join(dataset_path, "{}-train-labels.pkl".format(dataset)), "wb") as f:
        pickle.dump(train_labels, f)
    
    test_features = get_features(os.path.join(dataset_path, "x_test.mat"), features_dim)
    test_features=np.transpose( test_features,(3,0,1,2))
    test_features=test_features.reshape(2000, -1)
    print("Shape of train_features after final reshaping:", train_features.shape)
    test_labels = get_labels(os.path.join(dataset_path, "y_test.mat"), features_dim, labels_dim)
    print("Shape of train_features after final reshaping:", test_labels.shape)
    with open(os.path.join(dataset_path, "{}-test-features.pkl".format(dataset)), "wb") as f:
        pickle.dump(test_features, f)
    with open(os.path.join(dataset_path, "{}-test-labels.pkl".format(dataset)), "wb") as f:
        pickle.dump(test_labels, f)
    
    validation_features = get_features(os.path.join(dataset_path, "x_valid.mat"), features_dim)
    validation_features=np.transpose( validation_features,(3,0,1,2))
    validation_features=validation_features.reshape(2000, -1)
    validation_labels = get_labels(os.path.join(dataset_path, "y_valid.mat"), features_dim, labels_dim)
    with open(os.path.join(dataset_path, "{}-valid-features.pkl".format(dataset)), "wb") as f:
        pickle.dump(validation_features, f)
    with open(os.path.join(dataset_path, "{}-valid-labels.pkl".format(dataset)), "wb") as f:
        pickle.dump(validation_labels, f)


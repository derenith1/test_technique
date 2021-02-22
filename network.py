import numpy as np
from sklearn.preprocessing import OneHotEncoder
from joblib import load
from src.training import scale_test
from itertools import chain
import os


def return_prediction(data, clf_filename='clf_rf'):
    """
        Returns the predicted category of an object by using a trained network.

                Parameters:
                        data (list): List of the 4 parameters (float or integer) of the object. These parameters
                        are the height, the width, the depth and the weight of the concerned object.
                        Raise a Value_Error exception if the 4 parameters describe abnormal object dimensions.
                        clf_filename (str): The file name of a trained classifier without its extension.
                        Default extension is a joblib file. (Default : 'clf_rf')

                Returns:
                        flatten_list (str): Predicted category of the object. Can be either 'mlp', 'deco' or 'meuble'.
        """

    # We put the data into a nested list
    inputs = [data]

    # We scale the data by retrieving the scaler computed on the training dataset
    print("directory", os.getcwd())

    try:
        test_scaled = scale_test(inputs, filename='src/scaler_mdm.joblib')

    except FileNotFoundError:
        test_scaled = scale_test(inputs, filename='../src/scaler_mdm.joblib')

    # We load the trained classifier and make the prediction on the data
    try:
        clf = load('src/' + clf_filename + '.joblib')
    except FileNotFoundError:
        clf = load('../src/' + clf_filename + '.joblib')

    print(test_scaled)
    prediction = clf.predict(test_scaled)

    # We load the encoder to decode the prediction into a string
    load_onehot = OneHotEncoder()
    load_onehot.drop_idx_ = None
    try:
        load_onehot.categories_ = np.load('src/classes_onehot.npy', allow_pickle=True)
    except FileNotFoundError:
        load_onehot.categories_ = np.load('../src/classes_onehot.npy', allow_pickle=True)

    # Raises ValueError Exception if the values are outliers
    try:
        str_pred = load_onehot.inverse_transform(prediction)
    except ValueError:
        return "Values are too different from training dataset"

    # Return the category predicted
    flatten_list = list(chain.from_iterable(str_pred))
    print("Catégorie prédite : ", flatten_list[0])

    return flatten_list[0]


if __name__ == "__main__":
    # Simple test to run a prediction without the API
    test = [30, 30, 5, 3.15]
    str = return_prediction(test)

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from joblib import load
from src.training import scale_test
from itertools import chain


def return_prediction(data, clf_filename='clf_rf'):
    inputs = [data]
    test_scaled = scale_test(inputs, filename='src/scaler_mdm.joblib')
    clf = load('src/'+clf_filename+'.joblib')
    print (test_scaled)
    prediction = clf.predict(test_scaled)
    load_onehot = OneHotEncoder()
    load_onehot.drop_idx_ = None
    load_onehot.categories_ = np.load('src/classes_onehot.npy', allow_pickle=True)

    try:
        str_pred = load_onehot.inverse_transform(prediction)
    except ValueError:
        return "Values are too different from training dataset"

    flatten_list = list(chain.from_iterable(str_pred))
    print("Catégorie prédite : ", flatten_list[0])
    return flatten_list[0]

if __name__ == "__main__":
    test = [30, 30, 5, 3.15]
    str = return_prediction(test)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer, LabelEncoder
from joblib import dump, load
from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from itertools import chain
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')


# from keras.models import Sequential
# from keras.layers import Dense
# from keras.callbacks import TensorBoard
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=DeprecationWarning)


def read_data(filename='dataset.csv', delete_outliers = False):

    mdm_data = pd.read_csv(filename)
    mdm_data.head()

    if delete_outliers :
        z = np.abs(stats.zscore(mdm_data.iloc[:, :4]))
        threshold = 3
        row_outliers, _ = np.where(z > threshold)
        new_mdm = mdm_data.drop(row_outliers)
        print("Number of values removed : ", mdm_data.shape[0] - new_mdm.shape[0])
        mdm_data = new_mdm

    n_obs, n_var = mdm_data.shape
    X = mdm_data.iloc[:, :n_var - 1]
    Y = mdm_data['activity']

    # List of the unique names of Y. Here is ['mlp', 'deco', 'meuble'] are the 3 categories that we want to classify
    names = list(set(Y))
    enc = OneHotEncoder()
    Y_enc = enc.fit_transform(Y[:, np.newaxis]).toarray()

    n_features = X.shape[1]
    n_classes = len(names)

    np.save('classes_onehot.npy', enc.categories_)
    load_onehot = OneHotEncoder()
    load_onehot.categories_ = np.load('classes_onehot.npy', allow_pickle=True)

    return X, Y_enc


# Visualize the data sets


def create_scaler(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    dump(scaler, 'scaler_mdm.joblib')
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def scale_test(data, filename):
    scaler = load(filename)
    test_scaled = scaler.transform(data)
    return test_scaled


def return_acc_clf(X_scaled, Y_train, X_test_scaled, Y_test, number_estimators=250, filename='clf_rf'):
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=number_estimators)
    clf.fit(X_scaled, Y_train)
    y_pred = clf.predict(X_test_scaled)

    dump(clf, filename + '.joblib')  # save the model4
    accuracy = metrics.accuracy_score(Y_test, y_pred)
    print("Accuracy of the model : ", accuracy)
    return accuracy


def return_acc_clf_tree(X_scaled, Y_train, X_test_scaled, Y_test, filename='clf_tree'):
    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(X_scaled, Y_train)
    y_pred = tree_clf.predict(X_test_scaled)
    dump(tree_clf, filename + '.joblib')  # save the model4
    accuracy = metrics.accuracy_score(Y_test, y_pred)
    print("Accuracy of the model : ", accuracy)
    return accuracy

def create_dict(Y):
    dict_activity = {
        "mlp": [0., 0., 1.],
        "meuble": [0., 1., 0.],
        "deco": [1., 0., 0.]
    }

    Y_enc = []
    for i in range(len(Y)):
        Y_enc.append(dict_activity[Y[i]])

    return Y_enc


def prediction_dict(dict_activity, prediction):
    for keys in dict_activity:
        if np.argmax(prediction) == np.argmax(dict_activity[keys]):
            str_pred = keys
    return str_pred


def get_confusion_matrix(Y_test, y_pred):
    load_onehot = OneHotEncoder()
    load_onehot.categories_ = np.load('classes_onehot.npy', allow_pickle=True)
    y_dec_pred = list(chain.from_iterable(load_onehot.inverse_transform(y_pred)))
    y_dec_test = list(chain.from_iterable(load_onehot.inverse_transform(Y_test)))
    confmat = confusion_matrix(y_dec_test, y_dec_pred, labels=["mlp", "deco", "meuble"])

    return confmat


def scatter_plot(X, Y, classes_names):
    feature_names = X.columns
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    for features in classes_names:
        X_plot = X[Y == features]
        plt.plot(X_plot.iloc[:, 0], X_plot.iloc[:, 1], linestyle='none', marker='o', label=features)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.axis('equal')
    plt.legend();

    plt.subplot(1, 2, 2)
    for features in classes_names:
        X_plot = X[Y == features]
        plt.plot(X_plot.iloc[:, 2], X_plot.iloc[:, 3], linestyle='none', marker='o', label=features)
    plt.xlabel(feature_names[2])
    plt.ylabel(feature_names[3])
    plt.axis('equal')
    plt.legend();


#### KERAS ####

# def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
#         # Create model
#     model = Sequential(name=name)
#     for i in range(n):
#         model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
#     model.add(Dense(output_dim, activation='softmax'))
#
#         # Compile model
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model
#
#
# n_neurons = 8
# n_dense_layer = 4
# models = create_custom_model(n_features, n_classes, n_neurons, n_dense_layer, 'model_mdm')
#
#
# history_dict = {}
#
# # TensorBoard Callback
# cb = TensorBoard()
#
# history_callback = model.fit(X_train, Y_train,
#                                 batch_size=5,
#                                 epochs=150,
#                                 verbose=0,
#                                 validation_data=(X_test, Y_test),
#                                 callbacks=[cb]
#                             )
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# history_dict[model.name] = [history_callback, model]
#
#
# fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
#
# for model_name in history_dict:
#     val_accurady = history_dict[model_name][0].history['val_accuracy']
#     val_loss = history_dict[model_name][0].history['val_loss']
#     ax1.plot(val_accurady, label=model_name)
#     ax2.plot(val_loss, label=model_name)
#
# ax1.set_ylabel('validation accuracy')
# ax2.set_ylabel('validation loss')
# ax2.set_xlabel('epochs')
# ax1.legend()
# ax2.legend();

if __name__ == "__main__":
    X, Y_enc = read_data(filename='data/dataset.csv', delete_outliers=False)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_enc, test_size=0.22, random_state=2)
    X_train_scaled, X_test_scaled = create_scaler(X_train, X_test)
    acc = return_acc_clf(X_train_scaled, Y_train, X_test_scaled, Y_test)
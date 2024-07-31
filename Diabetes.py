import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def DiabetesKNN():

    diabetes = pd.read_csv('diabetes.csv')

    print("Column of Dataset")
    print(diabetes.columns)

    print("First 5 records of the dataset")
    print(diabetes.head())

    print("Dimension of diabetes data: {}".format(diabetes.shape))

    # Select features (X) and target (y)
    X = diabetes.drop(columns=['Outcome'])
    y = diabetes['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, random_state=66)

    training_accuracy = []
    test_accuracy = []

    # Try n_neighbors from 1 to 10
    neighbors_settings = range(1, 11)

    for n_neighbors in neighbors_settings:
        # Build the model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, Y_train)

        # Record training set accuracy
        training_accuracy.append(knn.score(X_train, Y_train))

        # Record test set accuracy
        test_accuracy.append(knn.score(X_test, Y_test))

    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.savefig('knn_compare_model')
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, Y_train)

    print('Accuracy of the kNN classifier on training set: {:.2f}'.format(knn.score(X_train, Y_train)))
    print('Accuracy of KNN classifier on test set: {:.2f}'.format(knn.score(X_test, Y_test)))

def main():
    print("---------------------------------------------------------------------")
    print("                DIABETES PREDICTION USING KNN ALGORITHM              ")
    print("---------------------------------------------------------------------")

    DiabetesKNN()

if __name__ == "__main__":
    main()

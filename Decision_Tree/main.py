import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


def evaluate_model(training_file, testing_file):
    training_data = pd.read_csv(training_file, sep='[,,\s]', header=None, engine='python')
    testing_data = pd.read_csv(testing_file, sep='[,,\s]', header=None, engine='python')

    features_train = training_data.values

    training_samples, features = training_data.shape
    testing_samples, _ = testing_data.shape

    features_test = testing_data.values
    labels_test = testing_data.iloc[:, testing_data.shape[1] - 1].values
    labels_train = training_data.iloc[:, training_data.shape[1] - 1].values

    # Initialize model, train it, and make predictions
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)

    # Output the results
    print("-" * 24 + " DECISION TREE EVALUATION " + "-" * 24)
    print("⁕ TRAINING FILE: %s, WITH %d SAMPLES" % (training_file, training_samples))
    print("⁕ TESTING FILE: %s, WITH %d SAMPLES" % (testing_file, testing_samples))
    print("⁕ ACCURACY SCORE: %d%%" % (metrics.accuracy_score(labels_test, predictions) * 100))
    print("⁕ CONFUSION MATRIX:\n", metrics.confusion_matrix(labels_test, predictions))
    print("⁕ CLASSIFICATION REPORT:\n", metrics.classification_report(labels_test, predictions))
    print("-" * 24 + " END OF EVALUATION " + "-" * 24)


if __name__ == '__main__':
    evaluate_model('let.trn', 'let.tst')
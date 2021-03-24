from sklearn import naive_bayes, model_selection
import numpy as np


def TEST_naiveBayesMultinomial(featureData, targetData):
    X = np.array(featureData)  # Convert the feature mask to a numpy array [item, feature mask]
    y = np.array(targetData)

    XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=0.25, random_state=None)

    mNB = naive_bayes.ComplementNB()
    mNB.fit(XTrain, yTrain)
    predictions = mNB.predict(XTest)
    # predictionProbs = mNB.predict_log_proba(XTest)
    return [predictions, yTest]

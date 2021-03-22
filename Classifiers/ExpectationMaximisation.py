from sklearn import mixture, model_selection


def TEST_gaussianMixture(X, y):
    XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)

    gmm = mixture.GaussianMixture()
    gmm.fit(XTrain, yTrain)
    predictions = gmm.predict(XTest)

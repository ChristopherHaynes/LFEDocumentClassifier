from sklearn import cluster, model_selection


def TEST_kmeans(X, y):
    XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)

    km = cluster.KMeans(n_clusters=24, random_state=42)
    km.fit(XTrain)
    kmLabels = km.labels_
    predictions = km.predict(XTest)

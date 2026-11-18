"""Model definitions: paper-replication baselines and a K-Means classifier wrapper."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from .preprocessing import make_pipeline


class KMeansClassifier(BaseEstimator, ClassifierMixin):
    """K-Means with 2 centroids used as a classifier (paper Sec. 4.4).

    The paper does NOT say how the two clusters are mapped to labels. Our explicit
    choice: the cluster with the higher training-set positive rate is 'class 1'.
    (Using raw cluster ids would make the label assignment arbitrary.)
    Score = distance to the negative centroid - distance to the positive centroid.
    """

    def __init__(self, n_clusters=2, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        self.km_ = KMeans(self.n_clusters, n_init=self.n_init, random_state=self.random_state).fit(X)
        rates = [y[self.km_.labels_ == c].mean() if (self.km_.labels_ == c).any() else 0.0
                 for c in range(self.n_clusters)]
        self.pos_cluster_ = int(np.argmax(rates))
        self.neg_cluster_ = 1 - self.pos_cluster_
        self.cluster_pos_rates_ = rates
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        d = self.km_.transform(np.asarray(X))
        return d[:, self.neg_cluster_] - d[:, self.pos_cluster_]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def baseline_estimators(seed, scale_all):
    """Paper-replication baselines (sklearn defaults unless the paper states otherwise).

    scale_all=False -> 'paper-faithful': only the MLP is scaled (the paper only
                       mentions MinMaxScaler for the MLP).
    scale_all=True  -> control variant: every model gets MinMaxScaler, so the effect
                       of scaling can be separated from the effect of tuning.
    """
    specs = {
        "Perceptron": (Perceptron(random_state=seed), False),
        "Logistic Regression": (LogisticRegression(random_state=seed), False),
        "K-Means": (KMeansClassifier(n_clusters=2, random_state=seed), False),
        "KNeighbors": (KNeighborsClassifier(), False),
        "SVM": (SVC(kernel="linear"), False),
        # Paper: relu, adam, constant LR, L2 penalty 1e-4, "100 hidden layers"
        # -> interpreted as hidden_layer_sizes=(100,) (sklearn default / one layer of 100 units)
        "MLP": (MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
                              alpha=1e-4, learning_rate="constant", random_state=seed), True),
    }
    return {n: make_pipeline(e, scaled or scale_all) for n, (e, scaled) in specs.items()}

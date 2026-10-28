"""Classical Machine Learning Classifiers module.

Implements the classifiers evaluated in the manuscript:
1. Linear SVM
2. Quadratic SVM (degree 2 polynomial)
3. Cubic SVM (degree 3 polynomial)
4. Fine Gaussian SVM (RBF kernel, gamma=0.5)
5. Medium Gaussian SVM (RBF kernel, gamma='scale')
6. Weighted KNN (k=10, distance-weighted)
7. Ensemble Bagged Trees (Random Forest with 100 trees)
8. Neural Network Classifier (MLPClassifier, 100 hidden units)

Guarantees:
- Standardization (StandardScaler) is fit strictly on training fold data.
- Deterministic random state (SEED = 42).
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

SEED = 42

def get_classifier(classifier_name):
    """
    Instantiate specified classifier model with calibrated probability outputs.
    """
    name = classifier_name.lower().replace("-", " ").replace("_", " ").strip()
    
    if name in ["linear svm", "linearsvm"]:
        return SVC(kernel="linear", probability=True, random_state=SEED)
        
    elif name in ["quadratic svm", "quadraticsvm"]:
        return SVC(kernel="poly", degree=2, probability=True, random_state=SEED)
        
    elif name in ["cubic svm", "cubicsvm"]:
        return SVC(kernel="poly", degree=3, probability=True, random_state=SEED)
        
    elif name in ["fine gaussian svm", "finegaussiansvm", "fine rbf"]:
        # Fine Gaussian: high gamma (narrow kernel width)
        return SVC(kernel="rbf", gamma=0.5, probability=True, random_state=SEED)
        
    elif name in ["medium gaussian svm", "mediumgaussiansvm", "medium rbf"]:
        # Medium Gaussian: standard scale gamma
        return SVC(kernel="rbf", gamma="scale", probability=True, random_state=SEED)
        
    elif name in ["weighted knn", "weightedknn", "knn"]:
        return KNeighborsClassifier(n_neighbors=10, weights="distance")
        
    elif name in ["ensemble bagged tree", "bagged tree", "bagged trees", "random forest"]:
        return RandomForestClassifier(n_estimators=100, max_features=None, random_state=SEED)
        
    elif name in ["neural network classifier", "neural network", "mlp"]:
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=SEED)
        
    else:
        raise ValueError(f"Unknown classifier name: {classifier_name}")

ALL_CLASSIFIER_NAMES = [
    "Linear SVM",
    "Quadratic SVM",
    "Cubic SVM",
    "Fine Gaussian SVM",
    "Medium Gaussian SVM",
    "Weighted KNN",
    "Ensemble Bagged Tree",
    "Neural Network Classifier"
]

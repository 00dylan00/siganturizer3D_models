"""Test Run Model

The objective of this script is to train a model on a single target.

Structure:
    1. Imports, Variables, Functions
    2. Load Data
    3. Define Cross Validation
    4. Define Model
    5. Train Model
    6. Evaluate Model
    7. Store Results
"""

# 1. Imports, Variables, Functions
# imports
import sklearn
import h5py, numpy as np, os, pickle
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
import sys
from tqdm import tqdm
import logging
from sklearn.metrics import average_precision_score
import joblib  # To save the model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# variables
feature_variable = str(sys.argv[1])
print(feature_variable)
feature_variables = [
    "a1.signaturizer",
    "b4.signaturizer",
    "b4.signaturizer3d",
    "ecfp4.useChirality",
    "ecfp4",
    "mapc",
    "unimol"
]

assert feature_variable in feature_variables, "ERR - Feature variable not found"

target = str(sys.argv[2])
iteration = int(sys.argv[3])

assert target in ["top", "bottom"], "ERR - Target not found"

features_file_path = os.path.join("..", "data", "features", f"{feature_variable}.h5")
compounds_file_path = os.path.join("..", "data", "target_compounds")

results_path = os.path.join("..", "output", "results", feature_variable)
models_path = os.path.join("..", "output", "models", feature_variable)


if not os.path.exists(results_path):
    os.mkdir(results_path)

if not os.path.exists(models_path):
    os.mkdir(models_path)


# functions


# 2. Load Data
# load list of indexes
if target == "top":
    with open(
        os.path.join("..", "data", "B4.001", "top_1_percent_targets.pkl"), "rb"
    ) as f:
        target_idxs = pickle.load(f)
    target_idx = target_idxs[iteration]

elif target == "bottom":
    with open(
        os.path.join("..", "data", "B4.001", "bottom_99_percent_targets.pkl"), "rb"
    ) as f:
        target_idxs = pickle.load(f)
    target_idx = target_idxs[iteration]

logging.info(f"Target index: {target_idx}")

# load classes
logging.info(f"Loading data from {compounds_file_path}")

with open(os.path.join(compounds_file_path, f"target_{target_idx}.pkl"), "rb") as f:
    d_classes = pickle.load(f)

positive_idxs = d_classes["positive_idxs"]
negative_idxs = d_classes["negative_idxs"]

positive_iks = d_classes["positive_iks"]
negative_iks = d_classes["negative_iks"]

logging.info(
    f"Positive samples: {len(positive_idxs)}, Negative samples: {len(negative_idxs)}"
)

# load features
logging.info(f"Loading features from {features_file_path}")

with h5py.File(features_file_path, "r") as f:
    positive_V = f["V"][positive_idxs]
    negative_V = f["V"][negative_idxs]
    positive_iks_loaded = f["inchikeys"][positive_idxs].astype("<U")
    negative_iks_loaded = f["inchikeys"][negative_idxs].astype("<U")

logging.info(f"Loaded features shape: {positive_V.shape}, {negative_V.shape}")

# check if iks match
assert (positive_iks_loaded == positive_iks).all, "ERR iks do not match"
assert (negative_iks_loaded == negative_iks).all, "ERR iks do not match"


# define data and labels
X_data = np.concatenate([positive_V, negative_V], axis=0)
y_data = np.concatenate([np.ones(len(positive_V)), np.zeros(len(negative_V))], axis=0)

logging.info(
    f"Merged positive & negative class data successfully: {X_data.shape}, {y_data.shape}"
)

logging.info(f"Size X of data {sys.getsizeof(X_data) / 1e6} MB")

# 3. Define Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Train and Evaluate Model
results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": [],
    "confusion_matrix": [],
    "roc_auc": [],
    "auprc": [],
    "mcc": [],
    "balanced_accuracy": [],
    "random_auprc": [],
    "target_idx": [],
    "n_positives_test": [],
    "n_positives_total": [],
    "fold":[]
}


# Add storage for train and test indexes
indexes = {
    "train_indexes": [],
    "test_indexes": []
}

fold = 0
for train_index, test_index in tqdm(skf.split(X_data, y_data)):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    # Check for positive class in both train and test sets
    assert np.any(y_train == 1), "Train fold does not have any positive samples"
    assert np.any(y_test == 1), "Test fold does not have any positive samples"


    # Save train and test indexes for this fold
    indexes["train_indexes"].append(train_index)
    indexes["test_indexes"].append(test_index)


    # Train the model
    model = XGBClassifier()

    # 5. Define Model
    model.fit(X_train, y_train)

    # 6. Evaluate Model
    # Predict and evaluate
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # Append metrics to results dictionary
    results["accuracy"].append(accuracy)
    results["precision"].append(precision)
    results["recall"].append(recall)
    results["f1_score"].append(f1)
    results["confusion_matrix"].append(conf_matrix)
    results["roc_auc"].append(roc_auc)
    results["auprc"].append(auprc)
    results["mcc"].append(mcc)
    results["balanced_accuracy"].append(balanced_acc)
    results["random_auprc"].append(np.sum(y_test) / len(y_test))
    results["n_positives_test"].append(np.sum(y_test))
    results["n_positives_total"].append(np.sum(y_data))
    results["target_idx"].append(target_idx)
    results["fold"].append(fold)

    fold += 1
    logging.info(f"Nº of positive samples in test: {np.sum(y_test)}")
    logging.info(
        f"Fold {fold} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )

# 6. Store Results
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(results_path, f"results.target_idx_{target_idx}.csv"), index=False
)
logging.info(f"Results stored in {results_path}")



# Save train and test indexes
indexes_path = os.path.join(results_path, f"indexes.target_idx_{target_idx}.pkl")
with open(indexes_path, 'wb') as f:
    pickle.dump(indexes, f)

logging.info(f"Results and indexes stored in {results_path}")

# Train the model on the entire dataset
final_model = XGBClassifier()
final_model.fit(X_data, y_data)

# Save the trained model
model_path = os.path.join(results_path, f"final_model.target_idx_{target_idx}.pkl")
joblib.dump(final_model, model_path)

logging.info(f"Final model stored at {model_path}")
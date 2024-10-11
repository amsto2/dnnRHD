"""
Created on Fri Feb  2 02:37:06 2024

@author: Amsalu Tomas Chuma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def get_score_after_permutation(model, X, y, curr_feat):
    """return the score of model when curr_feat is permuted"""
    X_permuted = X.copy()
    col_idx = list(X.columns).index(curr_feat)
    # permute one column
    X_permuted.iloc[:, col_idx] = np.random.permutation(X_permuted[curr_feat].values)
    permuted_score = model.score(X_permuted, y)
    return permuted_score

def get_feature_importance(model, X, y, curr_feat):
    """compare the score when curr_feat is permuted"""
    baseline_score_train = model.score(X, y)
    permuted_score_train = get_score_after_permutation(model, X, y, curr_feat)
    # feature importance is the difference between the two scores
    feature_importance = baseline_score_train - permuted_score_train
    return feature_importance

def permutation_importance(model, X, y, n_repeats=10):
    """Calculate importance score for each feature."""
    importances = []
    for curr_feat in X.columns:
        list_feature_importance = []
        for n_round in range(n_repeats):
            list_feature_importance.append(
                get_feature_importance(model, X, y, curr_feat)
            )
        importances.append(list_feature_importance)
    return {
        "importances_mean": np.mean(importances, axis=1),
        "importances_std": np.std(importances, axis=1),
        "importances": importances,
    }

def plot_feature_importances(perm_importance_result, feat_name):
    """bar plot the feature importance"""
    fig, ax = plt.subplots()
    indices = perm_importance_result["importances_mean"].argsort()
    plt.barh(
        range(len(indices)),
        perm_importance_result["importances_mean"][indices],
        xerr=perm_importance_result["importances_std"][indices],
    )
    ax.set_yticks(range(len(indices)))
    _ = ax.set_yticklabels(feat_name[indices])
    
    

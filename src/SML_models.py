import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, cohen_kappa_score, recall_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import pairwise_distances
from skbio.stats.ordination import pcoa
from skbio import DistanceMatrix

class SML:
    def __init__(self, n_estimators=200, cv_splits=3):
        self.n_estimators = n_estimators
        self.cv_splits = cv_splits

    # -----------------------
    # Train RF and evaluate
    # -----------------------
    def random_forest_MDL(self, X, Y):
        """Train Random Forest and return model + metrics"""
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.35, random_state=0, stratify=Y, shuffle=True
        )

        RF_mdl = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=0
        )
        RF_mdl.fit(x_train, y_train)
        y_pred = RF_mdl.predict(x_test)

        model_results = {
            "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 3),
            "Kappa Score": round(cohen_kappa_score(y_test, y_pred), 3),
            "Recall Score": recall_score(y_test, y_pred, average=None).tolist()
        }

        return RF_mdl, model_results

    # -----------------------
    # Cross-validation metrics
    # -----------------------
    def choose_bi(self, X, Y):
        """Cross-validation to get mean F1 and ROC-AUC"""
        RF_mdl = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0)
        scoring = {
            "F1_score": make_scorer(f1_score, average='weighted'),
            "AUC_ROC": 'roc_auc_ovr'
        }
        skf = StratifiedKFold(n_splits=self.cv_splits, random_state=0, shuffle=True)
        RF_scores = cross_validate(RF_mdl, X, Y, scoring=scoring, return_estimator=True, cv=skf)

        results = {
            "Mean F1 Score": round(RF_scores["test_F1_score"].mean(), 3),
            "Mean AUC ROC": round(RF_scores["test_AUC_ROC"].mean(), 3)
        }

        print("Mean F1 Score:", results["Mean F1 Score"])
        print("Mean AUC ROC:", results["Mean AUC ROC"])

        return results

    def exp_var_svd(self, data):
        #ReDim
        U, S, Vt = np.linalg.svd(data, full_matrices=False)
        num_sv = np.arange(1, S.size+1)
        cum_var_explained = [np.sum(np.square(S[0:n])) / np.sum(np.square(S)) for n in num_sv]
        return cum_var_explained

    def dist_mult(self, X, dist_name):
        dist = pairwise_distances(X, metric=dist_name)
        dist_matrix = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist_matrix, 0)
        
        # Return centred distance matrix
        return DistanceMatrix(dist_matrix)

    def exp_var_pcoa(self, df):
        cos = self.dist_mult(df, 'cosine')
        pcoa_cos_result = pcoa(cos)
        eigenvalues = pcoa_cos_result.proportion_explained
        cum_var_explained = np.cumsum(eigenvalues)
        n_index_90 = np.argmax(cum_var_explained >= 0.9)
        return pcoa_cos_result, n_index_90

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline

class SVDHandler:
    def __init__(self, n_components_list, n_estimators=200, cv_splits=3):
        self.n_components_list = n_components_list
        self.n_estimators = n_estimators
        self.cv_splits = cv_splits

    def exp_var_svd(self, X):
        """Compute cumulative explained variance for SVD."""
        _, S, _ = np.linalg.svd(X, full_matrices=False)
        return np.cumsum(S**2) / np.sum(S**2)

    def choose_best_n_components(self, X, Y):
        """Select best number of SVD components using GridSearchCV + RF."""
        RF_mdl = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0)
        scoring = {
            "F1_score": "f1_weighted",
            "AUC_ROC": "roc_auc_ovr"
        }


        svd_pipeline = Pipeline([
            ('svd', TruncatedSVD()),
            ('classifier', RF_mdl)
        ])

        param_grid = {'svd__n_components': self.n_components_list}
        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=0)

        grid_svd = GridSearchCV(svd_pipeline, param_grid, cv=skf, scoring=scoring, refit=False)
        grid_svd.fit(X, Y)
        
        best_n_c = grid_svd.cv_results_['param_svd__n_components'][grid_svd.cv_results_['rank_test_F1_score'].argmin()]
        explained_var = round(self.exp_var_svd(X)[best_n_c],3)
        
        mean_f1 = grid_svd.cv_results_['mean_test_F1_score'].mean()
        mean_auc = grid_svd.cv_results_['mean_test_AUC_ROC'].mean()

        results = pd.DataFrame(
            [[best_n_c, explained_var, round(mean_f1, 3), round(mean_auc,3)]],
            columns=["Best NC", "exp var", "Mean F1 Score", "Mean AUC ROC"],
            index=["SVD"]
        )
        return results
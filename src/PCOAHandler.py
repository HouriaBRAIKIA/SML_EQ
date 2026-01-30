import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import pairwise_distances
from skbio.stats.ordination import pcoa
from skbio.stats.distance import DistanceMatrix

class PCoAHandler:
    def __init__(self, n_components_list, metric="cosine", n_estimators=200, cv_splits=3):
        self.n_components_list = n_components_list
        self.metric = metric
        self.n_estimators = n_estimators
        self.cv_splits = cv_splits

    # def compute_pcoa(self, X):
    #     """Compute PCoA from distance matrix"""
    #     dist = pairwise_distances(X, metric=self.metric)
    #     dist_matrix = 0.5 * (dist + dist.T)
    #     np.fill_diagonal(dist_matrix, 0)
    #     return pcoa(DistanceMatrix(dist_matrix))
    
    def dist_mult(self, X, dist_name):
        dist = pairwise_distances(X, metric=dist_name)
        dist_matrix = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist_matrix, 0)
        
        # Return centred distance matrix
        return DistanceMatrix(dist_matrix)
    
    def exp_var_pcoa(self, df):
        met = self.dist_mult(df, self.metric)
        pcoa_met_result = pcoa(met)
        eigenvalues = pcoa_met_result.proportion_explained
        cum_var_explained = np.cumsum(eigenvalues)
        n_index_90 = np.argmax(cum_var_explained >= 0.9)
        return pcoa_met_result, n_index_90

    def choose_best_n_components(self, X, Y):
        """Select best n_components for PCoA + RF"""
        RF_mdl = RandomForestClassifier(random_state=0, n_estimators=self.n_estimators)
        skf = StratifiedKFold(n_splits=self.cv_splits, random_state=0, shuffle=True)
        
        # Perform Grid Search for PCoA #########################################""
        scoring = { "F1_score": "f1_weighted",
                    "AUC_ROC": 'roc_auc_ovr'}
        
        pcoa_results = self.exp_var_pcoa(X)  
        X_pcoa_global = pcoa_results[0].samples
        best_f1_pcoa = 0
        best_auc_pcoa = 0
        best_n_c_pcoa = None
        for n_components in self.n_components_list:
            # Sélectionner les n_components colonnes
            X_reduced = X_pcoa_global.iloc[:, :n_components]
            cv_results = cross_validate(RF_mdl, X_reduced, Y, cv=skf, scoring=scoring)
        
           # Calculer le F1 moyen
            mean_f1_pcoa = cv_results["test_F1_score"].mean()
            mean_auc_pcoa = cv_results["test_AUC_ROC"].mean()
            
            # Mettre à jour le meilleur score si nécessaire
            if mean_f1_pcoa > best_f1_pcoa:
                best_f1_pcoa = mean_f1_pcoa
                best_n_c_pcoa = n_components
                best_auc_pcoa = mean_auc_pcoa

        explained_variance_pcoa = round(sum(pcoa_results[0].proportion_explained[:best_n_c_pcoa]),3)

        results = pd.DataFrame(
            [[best_n_c_pcoa, explained_variance_pcoa, round(best_f1_pcoa,3), round(best_auc_pcoa,3)]],
            columns=["Best NC", "exp var", "Mean F1 Score", "Mean AUC ROC"],
            index=["PCoA"]
        )
        return results, X_pcoa_global

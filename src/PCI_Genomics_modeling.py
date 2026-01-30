import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, recall_score
import seaborn as sns

class ModelTrainer:
    """
    A class to train a Random Forest model, evaluate its performance, and visualize feature importance
    using both model-based and SHAP explanations.

    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    title : str
        Title used for saving plots and labeling

    Methods
    -------
    train_random_forest():
        Trains a Random Forest classifier and evaluates its performance.

    mdl_importance(importances, feature_names, title):
        Plots the feature importance.

    explain_with_shap(rf_model, X):
        Generates SHAP-based visualizations and summary plots.
    """
    
    def __init__(self, X, y, title=""):
        self.X = X
        self.y = y
        self.title = title

    def train_random_forest(self):
        """
        Trains a Random Forest model and evaluates its performance on a test split.

        Returns
        -------
        rf_model : RandomForestClassifier
            Trained model
        performance : dict
            Dictionary containing F1, Kappa, and Recall scores
        """
        
        # Séparer les données en train et test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.35, random_state=0, stratify=self.y, shuffle=True)
        
        # Entraîner le modèle Random Forest
        rf_model = RandomForestClassifier(n_estimators=200, random_state=0)
        rf_model.fit(X_train, y_train)

        # Prédictions sur le jeu de test
        y_pred = rf_model.predict(X_test)

        # Calcul des scores de performance
        performance = {
            "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 3),
            "Kappa Score": round(cohen_kappa_score(y_test, y_pred), 3),
            "Recall Score": recall_score(y_test, y_pred, average=None).tolist()
        }

        self.mdl_importance(rf_model.feature_importances_, self.X.columns, "Feature Importance ( Random Forest function ) "+self.title)
        
        return rf_model, performance

    def mdl_importance(self, importances, feature_names, title):
        """
        Plot feature importances as a bar chart.

        Parameters
        ----------
        importances : array-like
            Importance scores
        feature_names : list or Index
            Names of the features
        title : str
            Plot title and filename prefix
        """
        
        # Créer un DataFrame pour trier et afficher les importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Trier les caractéristiques par importance décroissante
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        # importance_df_filtered = importance_df[importance_df['Importance'] > 0]
        importance_df_filtered = importance_df.head(20)

        # Afficher le diagramme en barres
        plt.figure(figsize=(18, 12))
        plt.title(title, fontsize=25)
        bars = plt.bar(importance_df_filtered['Feature'], importance_df_filtered['Importance'], color='skyblue')
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Variables", fontsize=25)
        plt.ylabel("Importance", fontsize=25)
        plt.tight_layout()
        
        # Ajouter les valeurs numériques sur les barres
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, 
                    yval + 0.01*yval,  # décalage de 1% au-dessus de la barre
                    f'{yval:.2f}', 
                    va='bottom', ha='center', fontsize=8)

        underscore_text = lambda text: "_".join(text.split())
        plt.savefig(underscore_text("../results/FI/"+title+".png"), dpi=300, bbox_inches='tight')
        plt.close()

    def explain_with_shap(self, rf_model, X):
        """
        Explains model predictions using SHAP values and generates summary plots.

        Parameters
        ----------
        rf_model : RandomForestClassifier
            Trained model to explain
        X : pd.DataFrame
            Feature dataset to compute SHAP values
        """
        
        # Créer un explainer SHAP pour le modèle
        explainer = shap.Explainer(rf_model)
        shap_values = explainer(X)
        shap_importance = np.abs(shap_values.values).mean(axis=0).mean(axis=1)
        
        self.mdl_importance(shap_importance, X.columns, "Feature Importance ( SHAP ) "+self.title)
        
        # Moyenne par feature et par classe (avec signe)
        mean_shap_per_class = np.abs(shap_values.values).mean(axis=0)  # shape: features x classes
        df_shap = pd.DataFrame(mean_shap_per_class, index=X.columns, columns=[f"Class_{i+2}" for i in range(mean_shap_per_class.shape[1])])
        
        # Top 5 features par importance moyenne
        top_features = df_shap.abs().mean(axis=1).sort_values(ascending=False).head(5).index
        df_shap_top = df_shap.loc[top_features]
        
        # Heatmap
        plt.figure(figsize=(12, max(6, 0.3*len(top_features))))
        sns.heatmap(df_shap_top, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Mean |SHAP|'})
        plt.title("SHAP Feature Importance per Class - " + self.title, fontsize=20)
        plt.ylabel("Features", fontsize=20)
        plt.xlabel("Classes", fontsize=20)
        plt.tight_layout()
        
        underscore_text = lambda text: "_".join(text.split())
        plt.savefig(underscore_text(f"../results/shap/shap_heatmap_{self.title}.png"), dpi=300)
        plt.close()

        return df_shap_top
        
        # SHAP summary plot for the class 2 (Good Quality)
        # shap.summary_plot(shap_values[:, :, 0], X, feature_names=X.columns, show=False)
        # plt.savefig("../results/shap/shap_1_"+self.title+".png", dpi=300, bbox_inches='tight')
        # plt.show()
        
        # shap.summary_plot(shap_values[:, :, 1], X, feature_names=X.columns, show=False)
        # plt.savefig("../results/shap/shap_2_"+self.title+".png", dpi=300, bbox_inches='tight')
        # plt.show()
        
        # shap.summary_plot(shap_values[:, :, 2], X, feature_names=X.columns, show=False)
        # plt.savefig("../results/shap/shap_3_"+self.title+".png", dpi=300, bbox_inches='tight')
        # plt.show()

        # shap.summary_plot(shap_values[:, :, 3], X, feature_names=X.columns, show=False)
        # plt.savefig("../results/shap/shap_4_"+self.title+".png", dpi=300, bbox_inches='tight')
        # plt.show()

        # plt.close()

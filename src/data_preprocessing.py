import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.normalization import Normalization

# Using R inside pythons 
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
from rpy2.robjects import pandas2ri

pandas2ri.activate()

pd.DataFrame.iteritems = pd.DataFrame.items
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DataPreprocessing:
    """
    A class for preprocessing OTU (Operational Taxonomic Units) data and associated environmental metadata.
    
    This class performs various preprocessing steps including:
    - Importing data
    - Filtering OTU and metadata tables
    - Normalizing OTU counts (TMM or CSS)
    - Matching OTU data with metadata
    - Converting ecological quality indices (AMBI, ISI, NSI, NQI1) into categorical status levels
    """
    
    def import_data(self, file_name):
        """Load OTU or metadata from a tab-delimited file."""
        data = pd.read_csv(file_name, delimiter='\t', decimal=',', encoding='latin1')
        return data

    def filter_otu(self, df, samples_list, filter_R, filter_C):
        """Filters OTU table by read depth and low abundance OTUs."""
        df = df.set_index("OTU_ID")

        #Drop samples columns that doesn't contain BI value
        otu_col_s = [m for m in df if m not in list(samples_list)]
        df = df.drop(otu_col_s, axis=1)

        df_T = df.T

        # get samples with sequencing depth above the total average of reads per samples
        row_sums = df_T.sum(axis=1)
        df_T = df_T.loc[row_sums > filter_R, :]

        # Removing rares OTUs -- 100 / 10 for bacteria and ciliates
        col_sums = df_T.sum(axis=0)
        df_T = df_T.loc[:, col_sums > filter_C]

        return df_T

    def filter_metadata(self, df):
        """Applies log transformation and standardization on selected environmental variables."""
        new_df = pd.DataFrame({})
        new_df["Distance_cage_gps"] = np.log1p(df["Distance_cage_gps"])
        new_df["Depth"] = np.log1p(df["Depth"])
        new_df["pH"] = df["pH"]

        # Normalisation des données
        scaler = StandardScaler()
        new_df[['Distance_cage_gps', 'Depth', 'pH']] = scaler.fit_transform(new_df[['Distance_cage_gps', 'Depth', 'pH']])

        # Ajout d'autres colonnes
        new_df[['AMBI', 'ISI', 'NSI', 'NQI1']] = df[['AMBI', 'ISI', 'NSI', 'NQI1']]
        new_df["samples_names"] = df.drop([m for m in df if "sample" not in m.lower()], axis=1)

        return new_df

    def drop_na(self, df):
        """Removes rows with missing values."""
        return df.dropna(how='any')

    def match_table(self, df1, df2):
        """Merges OTU and metadata tables on sample names."""
        df = df1.merge(df2, how='inner', on=['samples_names'])
        df = df.set_index("samples_names")
        return df

    def bi_to_status(self, df):
        """Converts continuous ecological indices (AMBI, ISI, etc.) to categorical status (1 to 5)."""
        df["AMBI"] = [{float(y) < 1.2: 1, 1.2 <= float(y) < 3.3: 2, 3.3 <= float(y) < 4.3: 3, 4.3 <= float(y) < 5.5: 4, float(y) >= 5.5: 5}[True] for y in df["AMBI"]]
        df["ISI"] = [{y < 4.5: 5, 4.5 <= y < 6.1: 4, 6.1 <= y < 7.5: 3, 7.5 <= y < 9.6: 2, y >= 9.6: 1}[True] for y in df["ISI"]]
        df["NSI"] = [{y < 10: 5, 10 <= y < 15: 4, 15 <= y < 20: 3, 20 <= y < 25: 2, y >= 25: 1}[True] for y in df["NSI"]]
        df["NQI1"] = [{y < 0.31: 5, 0.31 <= y < 0.49: 4, 0.49 <= y < 0.63: 3, 0.63 <= y < 0.82: 2, y >= 0.82: 1}[True] for y in df["NQI1"]]
        return df

    def preprocess_data(self, df_otu, df_metadata, filter_R, filter_C, normalize=""):
        """
        Full preprocessing of OTU and metadata.
        Returns:
        - Filtered OTU matched with metadata and converted to status.
        - Normalized OTU matched with metadata and converted to status (if normalization is applied).
    
        Parameters:
        - df_otu: OTU table
        - df_metadata: Metadata table
        - filter_R: Row filtering threshold
        - filter_C: Column filtering threshold
        - normalize: "TMM", "CSS", or "" (no normalization)
    
        Returns:
        - otu_status_filtered: Filtered OTU matched with metadata and converted to status
        - otu_status_normalized: Normalized OTU matched with metadata and converted to status (or None)
        """        
        
        # Process metadata
        metadata_f = self.filter_metadata(df_metadata)
        metadata = self.drop_na(metadata_f)

        # Filter OTU table
        otu_filtered = self.filter_otu(df_otu, metadata["samples_names"], filter_R, filter_C)
        otu_filtered["samples_names"] = otu_filtered.index

        # Match filtered OTU with metadata and convert to status
        otu_filtered_matched = self.match_table(otu_filtered, metadata)
        # otu_status_filtered = self.bi_to_status(otu_filtered_matched)

        otu_status_normalized = None
        if normalize:
            norm = Normalization()
            if normalize == "TMM":
                otu_normalized = norm.tmm_normalization(otu_filtered.drop(columns="samples_names"))
            elif normalize == "CSS":
                otu_normalized = norm.css_normalization(otu_filtered.drop(columns="samples_names"))
            else:
                raise ValueError(f"Unsupported normalization method: {normalize}")
    
            otu_normalized["samples_names"] = otu_filtered.index
            otu_normalized_matched = self.match_table(otu_normalized, metadata)
            otu_status_normalized = self.bi_to_status(otu_normalized_matched)
    
        return otu_filtered_matched, otu_status_normalized

import rpy2
import pandas as pd
import numpy as np

# Using R inside pythons
import rpy2.robjects.packages as rpackages

# Importing required R packages
utils = rpackages.importr('utils')
from rpy2.robjects import pandas2ri

# Activate pandas2ri to handle dataframes between R and Python
pandas2ri.activate()

class Normalization:
    """
    This class implements normalization methods for biological data using R.
    Methods:
        - tmm_normalization: TMM (Trimmed Mean of M-values) normalization.
        - css_normalization: CSS (Cumulative Sum Scaling) normalization.
    """
    
    def tmm_normalization(self, data):
        """
        Perform TMM normalization using R's edgeR package.

        Parameters:
        -----------
        data : pd.DataFrame
            A DataFrame with OTU counts (rows are samples, columns are OTUs).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the TMM-normalized values, log-transformed.
        """
        
        #Import data
        rpy2.robjects.r.assign("otu_table_T", data.T)
        
        rpy2.robjects.r('''
        library(edgeR)
        
        # Create a counts matrix (replace with your own data)
        counts <- otu_table_T
        
        # Create a DGEList object
        dge <- DGEList(counts = counts)
        
        # Perform TMM normalization
        dge <- calcNormFactors(dge, method = "TMM")
        
        # Access the normalized counts
        normalized_counts <- cpm(dge, log=TRUE)
        ''')
        #Return new data
        tmm = rpy2.robjects.r["normalized_counts"]
        return pd.DataFrame(tmm, index=data.T.index, columns=data.T.columns).T

    def css_normalization(self, data):
        """
        Perform CSS normalization using R's metagenomeSeq package.

        Parameters:
        -----------
        data : pd.DataFrame
            A DataFrame with OTU counts (rows are samples, columns are OTUs).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the CSS-normalized values, log-transformed.
        """
        
        #Import data
        rpy2.robjects.r.assign("otu_table_T", data)
        #CSS normalization with R metagenomSeq package
        rpy2.robjects.r('''
          # import package metagenomeSeq, containing CSS functions
          library(metagenomeSeq)
        
          # convert OTU table into data.frame-format
          OTU_read_count = as.data.frame(otu_table_T)
          print(dim(OTU_read_count))
          class(OTU_read_count)
        
          # convert OTU table into package format
          metaSeqObject = newMRexperiment(OTU_read_count)
          #metaSeqObject_CSS  = cumNorm( metaSeqObject , p=cumNormStatFast(metaSeqObject) )
          metaSeqObject_CSS  = cumNorm(metaSeqObject,p=cumNormStat(metaSeqObject))
          OTU_read_count_CSS = data.frame(MRcounts(metaSeqObject_CSS, norm=TRUE, log=TRUE))
          OTU_read_count_CSS
        
          ''')
        #Return new data
        return rpy2.robjects.r["OTU_read_count_CSS"]
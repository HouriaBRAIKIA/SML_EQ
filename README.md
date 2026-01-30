# SML_EQ

# Machine Learning for Marine Ecological Quality and Biodiversity Assessment

## 🧠 Description

This project accompanies two peer‑reviewed conference papers focusing on data preprocessing and machine learning for marine ecological quality and biodiversity assessment using environmental DNA (eDNA) and microbiome data.

### 📄 Associated Publications

**Efficient Data Preprocessing for Ecological Quality Assessment in Marine Environments**
Houria Braikia, Sana Ben Hamida, Marta Rukoz
ICAISE 2024 – The 6th International Conference on Artificial Intelligence and Smart Environments
Errachidia, Morocco, November 2024, pp. 355–360
DOI: [https://doi.org/10.1007/978-3-031-88304-0_49](https://doi.org/10.1007/978-3-031-88304-0_49)
HAL: [https://hal.science/hal-05059099](https://hal.science/hal-05059099)

---

**Random Forest Classifier for Marine Biodiversity Analysis**
Houria Braikia, Sana Ben Hamida, Marta Rukoz
2024 International Conference on Intelligent Systems and Computer Vision (ISCV)
Fez, France, May 2024, pp. 1–8
DOI: [https://doi.org/10.1109/ISCV60512.2024.10620111](https://doi.org/10.1109/ISCV60512.2024.10620111)
HAL: [https://hal.science/hal-04673568](https://hal.science/hal-04673568)

---

## 📊 Dataset

The datasets used in this project consist of marine eDNA metabarcoding data (OTU tables) combined with environmental metadata.

They are processed following the methodologies described in the associated publications, with a strong focus on:

* Data cleaning and filtering
* Normalization techniques
* Feature preparation for machine learning models

Raw data are not always provided directly in this repository and may be subject to access conditions described in the referenced articles.

---

## ⚙️ Installation Instructions

### 📦 Python

```bash
pip install -Iv rpy2==3.4.2
pip install pandas==1.5.3
pip install numpy matplotlib seaborn scikit-learn scikit-bio
```

⚠️ After installing **rpy2**, restart your Python kernel or runtime (important for Jupyter users).

---

### 📦 R packages (via rpy2)

In your Python script or notebook, execute:

```python
import rpy2.robjects as ro
ro.r('''
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("edgeR")
BiocManager::install("metagenomeSeq")
''')
```

These packages are required for OTU normalization:

* **edgeR**: TMM normalization
* **metagenomeSeq**: CSS normalization

---

## 🧪 How to Use

1. Place your OTU tables and metadata files in the `Data/` folder.
2. Open and run the main notebook:

```
notebooks/SML_EQ.ipynb
```

The notebook implements:

* Data preprocessing and normalization (TMM and CSS via R)
* Dimensionality reduction (SVD and PCoA)
* Random Forest classifier training

---

## 🧠 Technologies Used

* **Python**

  * pandas, numpy, scikit-learn
  * matplotlib, seaborn
  * scikit-bio, rpy2

* **R** (via rpy2)

  * edgeR
  * metagenomeSeq

---

## 🤝 Contributions

Contributions, suggestions, and issue reports are welcome.

---

## 📧 Contact

For more information, please contact:

**Houria Braikia**
📧 [houria.braikia@dauphine.psl.eu](mailto:houria.braikia@dauphine.psl.eu)

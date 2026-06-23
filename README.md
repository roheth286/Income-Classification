# 💼 Income Classification Model (AI/ML Task)

## 🎯 Project Goal  
This project builds a machine learning pipeline to predict whether an individual’s annual income exceeds \$50K/year based on census-style demographic and employment data. The workflow covers data loading, cleaning, encoding categorical variables, training multiple models, and comparing their performance using standard classification metrics.

---

## 🔬 Project Overview  

The modularized machine learning pipeline is structured as follows:

1. **Load and Clean Data (`src/data_cleaning.py`)**
   - Loads the local `adult.data` dataset.
   - Cleans missing values (`?` -> `NaN`) and drops missing rows & duplicate rows.
   - Aligns feature matrix and target labels.

2. **Feature Engineering & Split (`src/feature_encoding.py`)**
   - Encodes target variable (income) into `0` / `1`.
   - Converts binary variables using label encoding and multi-category variables using one-hot encoding.
   - Splits data into training (80%) and testing (20%) sets.

3. **Train Models (`src/model_training.py`)**
   - Trains three models: **Logistic Regression**, **Random Forest**, and **SVM (Support Vector Machine)**.

4. **Evaluate Model Performance (`src/model_evaluation.py`)**
   - Predicts and calculates classification metrics: **Accuracy**, **Precision**, **Recall**, **F1‑Score**, and **ROC‑AUC**.
   - Generates and plots a combined Receiver Operating Characteristic (ROC) curve.

---

## 📁 Project Structure  

```text
Income-Classification/
├── data/
│   └── adult.data             # Raw income census dataset
├── notebooks/
│   └── pipeline.ipynb         # Runs the pipeline cell-by-cell using %run magic commands
├── src/
│   ├── data_cleaning.py       # Stage 1: Load and clean raw dataset
│   ├── feature_encoding.py    # Stage 2: Feature encoding and splitting data
│   ├── model_training.py      # Stage 3: Train models (LR, Random Forest, SVM)
│   └── model_evaluation.py    # Stage 4: Evaluate performance and plot ROC curves
├── Income_Classification.ipynb  # Original reference notebook (untouched)
├── requirements.txt           # Python dependencies list
└── README.md                  # Documentation and setup guide (this file)
```

---

## 🚀 How to Run  

1. Navigate to the project root directory:
   ```bash
   cd Income-Classification
   ```
2. Install the necessary Python packages using the requirements file:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the interactive notebook:
   - Open `notebooks/pipeline.ipynb` in a Jupyter environment or VS Code.
   - Run the cells in sequence. Each cell will execute the corresponding script from the `src/` directory and print/plot outputs directly in the notebook.

---

## 📄 Notes  
- The models utilize standard parameters configured with a random state (`42`) to guarantee reproducibility.
- If the local `data/adult.data` file is missing, the cleaning script automatically falls back to downloading it directly from the official UCI Machine Learning repository.

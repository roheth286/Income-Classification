# 💼 Income Classification Model (AI/ML Task)

## 🎯 Project Goal  
This project builds a machine learning pipeline to predict whether an individual’s annual income exceeds \$50K/year based on census-style demographic and employment data. The workflow covers data loading, cleaning, encoding categorical variables, training multiple models, and comparing their performance using standard classification metrics.

---

## 🔬 Project Overview  

The notebook is structured into four main stages:

1. **Load, Clean, Encode Data**
   - Load the income dataset.
   - Clean the dataset and handle missing or inconsistent values.
   - Convert categorical variables into numeric form (encoding).
   - Align features and target into proper `X` (inputs) and `y` (labels).

2. **Train Multiple Models**
   - Train a **Logistic Regression** classifier.
   - Train a **Random Forest** classifier.
   - Train an **SVM (Support Vector Machine)** classifier.

3. **Evaluate Model Performance**
   - Compute and compare:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1‑Score**
     - **ROC‑AUC** and ROC curves

4. **Model Selection & Justification**
   - Compare evaluation metrics across all trained models.
   - Justify the choice of the best‑performing model based on quantitative results and task requirements (e.g., when to prioritize precision vs recall).

---

## 📁 Project Structure  

- `README.md` → Project description and usage instructions (this file).  
- `income_classification.ipynb` (or similar) → Main notebook containing data processing, model training, and evaluation code.

---

## 🚀 How to Run (Google Colab Recommended)  

1. Upload the `.ipynb` notebook to Google Drive.  
2. Open it with **Google Colaboratory** (`Right click → Open with → Google Colaboratory`) or via an “Open in Colab” link from GitHub.  
3. Run the cells in order:
   - Data loading, cleaning, and encoding.
   - Model training (Logistic Regression, Random Forest, SVM).
   - Evaluation and model comparison section.

Any additional package installations (e.g., `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`) are handled via `pip` commands inside the notebook if needed.

---

## 📄 Notes  

- The main focus is on building a clear, comparable set of baseline models for the income > \$50K classification task.  
- You can extend the notebook by adding more algorithms, tuning hyperparameters, or handling class imbalance if required.

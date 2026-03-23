# 🚢 Titanic Survival Prediction — Data Science Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Conda-Environment-green?logo=anaconda&logoColor=white" />
  <img src="https://img.shields.io/badge/Accuracy-80.45%25-brightgreen" />
  <img src="https://img.shields.io/badge/Status-Completed-blue" />
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Student Info](#-student-info)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Environment Setup](#-environment-setup)
- [Project Pipeline](#-project-pipeline)
- [Key Findings](#-key-findings)
- [Model Results](#-model-results)
- [Feature Importance](#-feature-importance)
- [Visualizations](#-visualizations)
- [Challenges & Lessons Learned](#-challenges--lessons-learned)
- [Full Report](#-full-report)
- [Technologies Used](#-technologies-used)
- [How to Run](#-how-to-run)
- [References](#-references)

---

## 📌 Project Overview

This project is a complete end-to-end Data Science analysis of the **Titanic disaster dataset**. The goal is to explore the factors that influenced passenger survival, build predictive machine learning models, and extract meaningful insights through data visualization.

The project covers the full data science pipeline:

- Data loading & exploration
- Data cleaning & preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model building & evaluation

> **Overall Result:** Achieved **80.45% prediction accuracy** using Random Forest, exceeding the 75% industry threshold, with a ROC-AUC score of **0.851**.

---

## 🎓 Student Info

| Field | Details |
|-------|---------|
| **Name** | Ibrahim Abdulmonem Ibrahim Abushouk |
| **Major** | Information Technology |
| **Course** | Introduction to Data Science |
| **GitHub** | [@ibrahimAbushouk](https://github.com/ibrahimAbushouk) |

---

## 📦 Dataset

- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- **Rows:** 891 passengers
- **Columns:** 12 features
- **Class Distribution:** 342 survivors (38.4%) · 549 non-survivors (61.6%)

| Column | Description |
|--------|-------------|
| `PassengerId` | Unique ID for each passenger |
| `Survived` | Survival status (0 = No, 1 = Yes) — **Target Variable** |
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Name` | Passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Ticket price |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C / Q / S) |

---

## 🗂 Project Structure

```
Titanic-Analysis/
│
├── 📓 _Titanic_Analysis.ipynb            # Main Jupyter Notebook (full analysis)
├── 📄 titanic_cleaned.csv                # Cleaned dataset (output after preprocessing)
├── 📋 README.md                          # Project documentation (this file)
└── 📑 Data_Science_Final_Report.pdf      # Full written report
```

---

## ⚙️ Environment Setup

This project was built using **Anaconda** with **Python 3**.

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or Miniconda installed
- Python 3.8+

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ibrahimAbushouk/titanic-analysis.git
cd titanic-analysis

# 2. Create a Conda environment
conda create -n titanic_env python=3.10

# 3. Activate the environment
conda activate titanic_env

# 4. Install required libraries
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 🔄 Project Pipeline

```
┌──────────────────────┐
│   1. Data Loading     │  ──→  Load CSV (891 rows × 12 columns)
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  2. Data Exploration  │  ──→  Shape, dtypes, .describe(), missing values check
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│   3. Data Cleaning    │  ──→  Age → median (28.0) | Embarked → mode ('S') | Cabin → dropped
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  4. Feature Eng.      │  ──→  FamilySize = SibSp + Parch + 1 | Encode Sex & Embarked
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  5. EDA & Visuals     │  ──→  6 comprehensive charts covering all key factors
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  6. ML Modeling       │  ──→  Logistic Regression | Decision Tree | Random Forest
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  7. Evaluation        │  ──→  Accuracy · Precision · Recall · F1 · ROC-AUC · Confusion Matrix
└──────────────────────┘
```

**Data Preprocessing Summary:**

| Feature | Missing Values | Strategy Applied |
|---------|---------------|-----------------|
| Age | 177 (19.87%) | Filled with median = 28.0 |
| Embarked | 2 (0.22%) | Filled with mode = 'S' |
| Cabin | 687 (77.10%) | Column dropped |

**Final Dataset:** 891 rows × 10 columns (9 features + 1 target), **zero missing values**.

---

## 🔍 Key Findings

### Survival by Gender
| Gender | Survived | Rate |
|--------|----------|------|
| Female | 233 / 314 | **74.20%** |
| Male | 109 / 577 | **18.89%** |

> Women were **3.9× more likely** to survive — reflecting the "women and children first" evacuation protocol.

### Survival by Passenger Class
| Class | Survived | Rate |
|-------|----------|------|
| 1st Class | 136 / 216 | **62.96%** |
| 2nd Class | 87 / 184 | **47.28%** |
| 3rd Class | 119 / 491 | **24.24%** |

> Clear socioeconomic gradient — wealth provided a measurable survival advantage.

### Survival by Age Group
| Age Group | Survival Rate |
|-----------|--------------|
| Children (0–12) | **54%** |
| Adults (19–35) | **37%** |
| Seniors (60+) | **22%** |

### Survival by Family Size
| Family Size | Survival Rate |
|-------------|--------------|
| Solo (1) | 30% |
| Small (2–4) | **72%** ✅ Optimal |
| Large (7+) | 0% |

### Fare Impact
- Survivors paid an average of **£48.40**
- Non-survivors paid an average of **£22.12**

### Correlation with Survival
| Feature | Correlation | Interpretation |
|---------|------------|----------------|
| Sex | -0.543 | Strong — being male reduced survival |
| Pclass | -0.338 | Moderate — lower class reduced survival |
| Fare | +0.257 | Moderate — higher fare increased survival |
| Age | -0.077 | Weak — older age slightly reduced survival |

---

## 🤖 Model Results

Data split: **80% training** (712 passengers) / **20% testing** (179 passengers) — stratified sampling.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 79.89% | 76.92% | 71.43% | 74.07% | 0.830 |
| Decision Tree | 78.21% | 75.00% | 67.86% | 71.23% | 0.800 |
| **Random Forest** ✅ | **80.45%** | **78.79%** | **73.21%** | **75.89%** | **0.851** |

### Confusion Matrix — Random Forest (Best Model)

|  | Predicted: Not Survived | Predicted: Survived |
|--|------------------------|---------------------|
| **Actual: Not Survived** | 103 ✅ | 17 ❌ |
| **Actual: Survived** | 18 ❌ | 41 ✅ |

> Correctly classified **144 out of 179** test passengers (**80.45%**).

---

## 📊 Feature Importance

Based on the Random Forest model, the top predictors of survival:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Sex | **44.2%** |
| 2 | Fare | **17.8%** |
| 3 | Age | **13.5%** |
| 4 | Pclass | **10.2%** |
| 5 | FamilySize | **8.1%** |

> The top 3 features account for **75.5%** of all model decisions, validating the historical "women and children first" policy and the impact of socioeconomic inequality.

---

## 📈 Visualizations

Six comprehensive visualizations were produced in the notebook:

1. **Grouped Bar Chart** — Survival rate by class and gender (1st-class females: 97% vs 3rd-class males: 13.5%)
2. **Histogram + Boxplot** — Age distribution by survival (survivors avg: 28.3 yrs vs 30.6 yrs)
3. **Correlation Heatmap** — Feature relationships identifying Sex, Pclass, and Fare as key predictors
4. **Line + Bar Chart** — Family size impact (optimal: 2–4 members)
5. **Violin Plot** — Fare distribution showing survivors had wider, higher fare range
6. **Model Comparison Dashboard** — All metrics side-by-side + Confusion Matrices + ROC Curves

---

## 🧩 Challenges & Lessons Learned

| Challenge | Solution | Lesson |
|-----------|----------|--------|
| Cabin column: 77% missing | Dropped the column entirely | Sometimes less data is better than bad data |
| Feature Engineering | Created `FamilySize` = SibSp + Parch + 1 | Domain knowledge guides better features |
| Model Selection | Compared 3 models systematically | Always compare multiple approaches |

**Most Valuable Lesson:**
> *"Data science requires both technical skills and domain understanding. High accuracy means little without interpreting why the model works and what it reveals about the underlying patterns."*

### Areas for Future Improvement
- Extract titles from passenger names (Mr., Mrs., Master) as additional features
- Implement hyperparameter tuning (GridSearchCV)
- Try advanced models (XGBoost, Neural Networks)
- Apply cross-validation for more robust evaluation
- Build an interactive dashboard for data exploration

---

## 📑 Full Report

A detailed written report documenting the entire analysis, methodology, and conclusions is available below:

> 📎 **[Click here to view the full project report — PDF](./Data_Science_Final_Report.pdf)**

The report covers: Executive Summary · Dataset Description · Methodology · EDA · Visualizations · Predictive Modeling · Results Interpretation · Reflection · Conclusions & Recommendations.

---

## 🛠 Technologies Used

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/-Python_3.x-blue?logo=python&logoColor=white) | Core programming language |
| ![Jupyter](https://img.shields.io/badge/-Jupyter_Notebook-orange?logo=jupyter&logoColor=white) | Interactive notebook environment |
| ![Anaconda](https://img.shields.io/badge/-Anaconda-green?logo=anaconda&logoColor=white) | Environment & package management |
| `pandas` | Data manipulation & analysis |
| `numpy` | Numerical computations |
| `matplotlib` | Static data visualizations |
| `seaborn` | Statistical data visualizations |
| `scikit-learn` | ML models, evaluation & preprocessing |

---

## ▶️ How to Run

```bash
# 1. Activate your Conda environment
conda activate titanic_env

# 2. Launch Jupyter Notebook
jupyter notebook

# 3. Open the main notebook
# → _Titanic_Analysis.ipynb

# 4. Run all cells
# Kernel → Restart & Run All
```

> ⚠️ The notebook downloads the Titanic dataset automatically from the web on first run — make sure you have an active internet connection.

---

## 📚 References

1. [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
2. [Scikit-learn Documentation](https://scikit-learn.org/)
3. [Pandas Documentation](https://pandas.pydata.org/)
4. Frey, B. S., et al. (2011). *Behavior under Extreme Conditions: The Titanic Disaster.* Journal of Economic Perspectives, 25(1), 209–222.
5. [Encyclopedia Titanica](https://www.encyclopedia-titanica.org/)

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/ibrahimAbushouk">Ibrahim Abushouk</a> · Python & Jupyter · Powered by Anaconda
</p>

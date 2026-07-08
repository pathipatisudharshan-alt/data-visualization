# 🚢 Titanic - Exploratory Data Analysis (EDA)

A complete exploratory data analysis of the Titanic passenger dataset, covering data cleaning, outlier detection, feature engineering, and categorical encoding — laying the groundwork for future machine learning models.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project explores the classic Titanic dataset to understand which factors — gender, age, class, fare, family size, and title — were most associated with passenger survival. Beyond basic cleaning, it includes outlier handling, engineered features, and encoding steps that prepare the data for machine learning (used in Month 2 of my Data Science learning roadmap).

---

## 🎯 Objectives

- Clean and preprocess raw passenger data
- Visually explore survival patterns across multiple variables
- Detect and treat outliers in numerical features
- Engineer new, more predictive features from raw columns
- Encode categorical variables for downstream ML use

---

## 🗂️ Dataset

- **Source:** [Titanic Dataset — Data Science Dojo](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Rows:** 891 passengers
- **Key columns:** `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Name`, `Cabin`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Label encoding |

---

## 🔧 Data Cleaning

- Filled missing `Age` values with the **median**
- Filled missing `Embarked` values with the **mode**
- Dropped `Cabin` (too many missing values to be useful)
- Verified no remaining nulls in critical columns

---

## 📊 Exploratory Visualizations

| Chart | What it shows |
|---|---|
| `titanic_eda_survival_count.png` | Overall survival distribution |
| `titanic_eda_count_by_gender.png` | Survival split by gender |
| `titanic_eda_hist.png` | Age distribution across passengers |
| `titanic_eda_scatter.png` | Fare vs Age, colored by survival |
| `titanic_eda_heatmap.png` | Correlation between numeric features |
| `titanic_eda_pairplot.png` | Pairwise relationships (Survived, Age, Fare, Pclass) |
| `titanic_eda_outliers.png` | Boxplots showing Fare & Age outliers |
| `titanic_eda_title.png` | Survival rate by extracted title (Mr/Mrs/Miss/Rare) |
| `titanic_eda_familysize.png` | Survival rate by family size |

---

## 🚨 Outlier Detection

Used the **IQR (Interquartile Range) method** to flag outliers in `Fare` and `Age`.

- Instead of dropping outlier rows (which would lose valuable survival signal from high-fare passengers), extreme `Fare` values were **capped at the 99th percentile** (`Fare_capped`).

---

## 🧩 Feature Engineering

New features created to boost predictive value for future ML models:

| Feature | Description |
|---|---|
| `Title` | Extracted from `Name` (Mr, Mrs, Miss, Master, Rare) |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | 1 if passenger traveled alone, else 0 |

These features revealed patterns not visible in the raw columns — for example, **title** captures social status and age-group signals in a single feature.

---

## 🔢 Categorical Encoding

To prepare the dataset for machine learning:

- **One-hot encoding** applied to `Sex`, `Embarked`, and `Title` → saved as `titanic_encoded.csv`
- **Label encoding** applied to `Sex` as an alternative compact representation

---

## 📈 Key Insights

- **Gender** was the strongest single predictor of survival — women survived at a substantially higher rate than men, consistent with the "women and children first" protocol.
- **Passenger class** mattered: 1st class passengers had notably higher survival rates than 3rd class.
- **Fare** correlates with class and survival — higher fares generally align with higher survival odds.
- **Family size** shows a non-linear effect — passengers traveling completely alone or in very large families had lower survival rates than those in small families (2–4 members).
- **Title** (a proxy for age/gender/social status) added a nuance that raw `Sex` and `Age` alone didn't fully capture.

*(Fill in your exact percentages here once you run the script — see the print statements in `titanic_eda.py`.)*

---

## 📁 Project Structure

```
titanic-eda/
│
├── titanic_eda.py                    # Full analysis script
├── titanic_cleaned_engineered.csv    # Cleaned + feature-engineered dataset
├── titanic_encoded.csv               # One-hot encoded dataset (ML-ready)
├── titanic_eda_*.png                 # All generated charts
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install pandas matplotlib seaborn scikit-learn

# Run the analysis
python titanic_eda.py
```

---

## 🔮 Next Steps

This EDA lays the foundation for **Month 2** of my Data Science roadmap:
- Train a Logistic Regression / Linear Regression baseline model
- Evaluate with accuracy, precision, recall
- Deploy a simple ML app (Render/Streamlit)

---

## 👤 Author

**Pathipati Sudharshan**
[LinkedIn](https://www.linkedin.com/in/pathipati-sudharshan-naidu-93b666413/) • [GitHub](https://github.com/pathipatisudharshan-alt)

---

## 📜 License

This project is open source and available for learning purposes.

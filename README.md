# Titanic Survival Prediction 🛳️

This project uses a machine learning classification model to predict whether a passenger survived the Titanic disaster, based on various features like age, gender, fare, and class.

## 🗂️ Dataset

The dataset used in this project is a variant of the well-known Titanic dataset. It includes:
- Passenger features such as `Age`, `Sex`, `Fare`, `Pclass`, etc.
- The target variable `Survived` (0 = No, 1 = Yes)

## 📊 Exploratory Data Analysis

The notebook performs:
- Missing value analysis
- Feature dropping (e.g., `Cabin`, `Ticket`, `Name`, `PassengerId`)
- Correlation heatmap to visualize relationships between features

## ⚙️ Preprocessing

- **Numerical features** (`Age`, `Fare`, etc.) are filled with median values and scaled using `StandardScaler`.
- **Categorical features** (`Sex`, `Embarked`) are filled with the most frequent value and encoded using one-hot encoding.
- Preprocessing is handled through a `ColumnTransformer` pipeline.

## 🧠 Models Used

The following classification models are trained and evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Each model is evaluated using:
- Accuracy Score
- Classification Report
- Confusion Matrix

## ✅ Results

ALL models show a accuracy score of 1 except Support Vector Machine': SVC() which has a accuracy score of 0.9881

## 🚀 How to Run

1. Install required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

2. Open the notebook:
    ```bash
    jupyter notebook titanic_classification_model.ipynb
    ```

3. Run all cells to train the models and view the results.

## 📁 Files

- `titanic_classification_model.ipynb` – Main Jupyter Notebook with all code
- `tested.csv` – Titanic dataset used for model training and evaluation

## 📌 Notes

- The dataset should be cleaned and preprocessed before training.
- Passenger names, tickets, and cabin information were not used due to high uniqueness and missingness.
- You can further improve the model by applying hyperparameter tuning and ensemble methods.

---


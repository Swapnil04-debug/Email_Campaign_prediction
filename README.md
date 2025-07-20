# 📧 Email Campaign Outcome Prediction (Multi-Class Classification)

This project aims to predict the outcome of email marketing campaigns using a classification model trained on engineered email features. The target variable, `Email_Status`, consists of **three classes**:
- `0`: No Action
- `1`: Clicked
- `2`: Converted

The solution includes **data preprocessing**, **feature engineering**, **class imbalance handling**, **model selection and tuning**, and **evaluation using appropriate classification metrics**.

---

## 🚀 Project Objectives

- Predict how customers respond to email campaigns
- Handle class imbalance (e.g., most users don't interact, very few convert)
- Identify key features that influence clicks and conversions
- Build a deployable model with explainability

---

## 📂 Dataset Overview

The dataset contains `60,123` records and `31` features, including:

- **Email metadata**: `Email_Type`, `Subject_Hotness_Score`, `Email_Campaign_Type`, `Time_Email_sent_Category`
- **Content metrics**: `Word_Count`, `Total_Links`, `Total_Images`, `Total_Content`
- **User behavior/location**: `Location_Time`, `Customer_Location_B` to `G`, `Location_Freq`
- **Engineered features**: `Hotness_X_Content`, `Image_to_Text_Ratio`, `Links_Per_Word`, etc.
- **Target**: `Email_Status` (0 = No Action, 1 = Clicked, 2 = Converted)

---

## 🔧 Preprocessing & Feature Engineering

- Outlier removal using **IQR method grouped by Email_Status**
- Categorical binning: `Hotness_Bin`, `Word_Count_Bin`
- Derived features: `Image_to_Text_Ratio`, `Content_Hotness_Ratio`, `Links_Per_Word`, etc.
- One-hot encoding for categorical/location columns
- Feature scaling using `StandardScaler`

---

## 🧠 Models Used

### 1. **Logistic Regression**
- Tuned using `RandomizedSearchCV`
- `class_weight='balanced'`
- Performance:  
  - Accuracy: ~83.14%  
  - Macro F1 Score: 0.4349  
  - ROC AUC: 0.6983

### 2. **XGBoost Classifier**
- SMOTEENN + class weighting
- Tuned with `learning_rate`, `max_depth`, `n_estimators`
- Balanced performance, good interpretability

### 3. **Voting Classifier (Soft Voting)**
- Combines Logistic Regression, Random Forest, and XGBoost
- Averaged performance, better than logistic but lower than ANN

---

### ⭐️ **4. Artificial Neural Network (Final Model)**

- Preprocessed with `StandardScaler`
- Class imbalance handled using **SMOTEENN**
- Trained on **20 numerical features** (after one-hot encoding and reduction)
- Used **Batch Normalization**, **Dropout**, and **ReLU** activation

| Metric               | Score     |
|----------------------|-----------|
| **Accuracy**         | **91–93%** (varies by seed) |
| **Macro Precision**  | **~0.78–0.82** |
| **Macro Recall**     | **~0.86–0.88** |
| **Macro F1 Score**   | **~0.82–0.84** |
| **ROC AUC Score**    | **~0.96+** |

✅ ANN provided **the best generalization** and **highest performance** on minority classes (Class 1 and Class 2), outperforming other models on macro metrics.

---

## 📊 Evaluation

- Used **macro-averaged precision, recall, and F1-score** due to class imbalance
- Plotted **confusion matrices**, **classification reports**, and **SHAP plots** (for interpretable models like XGBoost)
- Analyzed class-wise performance and misclassification patterns

---

## 🧪 Key Techniques Used

- **Imbalanced data handling**: SMOTEENN, class weights
- **Hyperparameter tuning**: RandomizedSearchCV for Logistic Regression
- **ANN regularization**: Dropout, BatchNorm
- **Feature importance**: SHAP + engineered features

---

## 📌 Conclusion

- The **ANN model was selected as the final model** due to its strong ability to generalize and handle non-linear patterns in imbalanced multi-class data.
- Custom feature engineering and imbalance correction played a major role in boosting performance.
- The project demonstrates how **deep learning can enhance predictive targeting in marketing workflows**.

---

## 📎 Future Improvements

- Integrate ANN into a Flask/Streamlit web app for UI-based predictions
- Try time-series models or session-based RNNs for sequential customer behavior
- Perform more advanced hyperparameter tuning (e.g., Optuna)

---

## ✅ Requirements

```bash
python>=3.8  
pandas  
numpy  
scikit-learn  
xgboost  
imblearn  
matplotlib  
seaborn  
tensorflow / keras  
shap  

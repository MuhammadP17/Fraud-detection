# Credit Card Fraud Detection

**Project Goal:** Identify fraudulent transactions in a dataset of European cardholders (September 2013). The dataset is highly imbalanced, with only 492 frauds out of 284,807 transactions (0.172%).

---

## 🤝 The Collaboration
We simulated a production workflow by splitting the project into **Data Engineering** and **Model Development**.

* **My Role (Data Engineering):** I managed the data pipeline—analyzing the PCA-transformed features, scaling the remaining data, and setting the performance baseline.
* **Partner's Role (Modelling):** My partner managed the model tuning—testing advanced algorithms (XGBoost, Isolation Forests) on the balanced data I prepared.

---

## 🛠 My Contribution: The Pipeline
I built the foundation that allowed the models to learn effectively:

* **EDA & Preprocessing:** * Identified that the "Time" and "Amount" features contained outliers. Applied **RobustScaler** to handle these extreme values without distorting the data distribution.
    * Compared correlation heatmaps between the original and balanced datasets to reveal hidden feature relationships.
* **Handling Imbalance:** * Implemented **Random Under Sampling (RUS)** to downsample the majority class.
    * Created a perfectly balanced (50/50) training set to prevent model bias, while keeping the Validation and Test sets in their original imbalanced state for realistic evaluation.
* **Rigorous Splitting:** * Designed a **Stratified Train/Val/Test split** (70/15/15). This ensured the rare fraud cases were evenly distributed across all three datasets, preventing data leakage.
* **Benchmarking:** * Trained a Logistic Regression baseline on the balanced data.
    * **Result:** Proved that a simple linear model could achieve high recall, setting a strong "floor" for my partner's complex models.

---

### 🤖 Partner's Contribution: Model Selection & Trade-off Analysis
My partner trained 6 different classifiers (including SVM, Gradient Boosting, and Naive Bayes) to find the best balance between catching fraud and minimizing false alarms.

* **The Conflict:** While **K-Nearest Neighbors (KNN)** achieved the highest raw F1 Score (0.83), we ultimately rejected it in favor of **Random Forest**.
* **The Decision:** We selected **Random Forest** as the final production model because KNN is computationally expensive at inference time (too slow for real-time transactions) and sensitive to noise in high-dimensional data.

**Performance Comparison:**

| Model | Precision | Recall | F1 Score | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9310** | **0.7297** | **0.8182** | **Selected (Best Balance)** |
| K-Nearest Neighbors | 0.9333 | 0.7568 | 0.8358 | *Rejected (Too slow)* |
| Naive Bayes | 0.0613 | 0.8378 | 0.1143 | *Rejected (Too many False Positives)* |
| Logistic Regression | 0.8814 | 0.7027 | 0.7820 | *Baseline* |

> **Key Insight:** Naive Bayes caught the most fraud (83% Recall) but had a Precision of only 6%, meaning it flagged thousands of innocent customers. Random Forest provided the stability and precision required for a banking environment.


---

## 💻 Tech Stack
* **Data Processing:** Pandas, NumPy
* **Machine Learning (Scikit-Learn):**
    * *Classifiers:* Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes, Logistic Regression.
    * *Metrics:* F1-Score, Precision-Recall, Confusion Matrix.
* **Model Persistence:** Joblib (saving the production model as `.pkl`).
* **Visualization:** Seaborn, Matplotlib.

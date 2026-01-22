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

* **EDA & Feature Analysis:** The dataset provided 28 anonymized PCA components (V1-V28). I identified that the only non-transformed features, **Time** and **Amount**, had different scales and required robust scaling to prevent model bias.
* **Handling Imbalance:** Implemented **Random Under Sampling** to address the extreme 0.17% fraud ratio, creating a balanced training set for the models.
* **Benchmarking:** Trained a Logistic Regression baseline. This set a "floor" for performance, ensuring that any complex model we built justified the computational cost.

---

## 🤖 Partner's Contribution: Advanced Modelling
* **Algorithm Testing:** Trained Random Forest and XGBoost classifiers on the pre-processed data.
* **Optimization:** Used GridSearch to optimize specifically for **Recall** (minimizing False Negatives), as missing a fraud case is more costly than flagging a legitimate one.

---

## 💻 Tech Stack
* **Python:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, SMOTE
* **Visualization:** Seaborn, Matplotlib

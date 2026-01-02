import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Import ML ready data (use balanced data samples)
X_train = pd.read_csv('ML Data/X_train.csv') 
X_test = pd.read_csv('ML Data/X_test.csv')
y_train = pd.read_csv('ML Data/y_train.csv')
y_test = pd.read_csv('ML Data/y_test.csv')

# Initialize model instances
models = {"Logistic Regression": LogisticRegression(),
          "Random Forest Classifier": RandomForestClassifier(),
          "Gradient Boosting Classifier": GradientBoostingClassifier(),
          "Gaussian Naive Bayes": GaussianNB(),
          "KNeighbors Classifier": KNeighborsClassifier(),
          "Support Vector Machine": svm.SVC()}

# Model training and performance evaluation
for name, model in models.items():
    model.fit(X_train, np.ravel(y_train))
    y_pred = model.predict(X_test)
    
    # Metrics output
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    # Confusion Matrices
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
# Saving best trained model based on metrics
best_model = models["Random Forest Classifier"].fit(X_train, np.ravel(y_train))
joblib.dump(best_model, 'fraud_det_model.pkl')

"""
Even though KNN showed a slightly better performance across the metrics, its shortfalls
make Random Forest a better choice. KNN can be computationally expensive for large datasets,
it struggles with high-dimensional datasets, and it can be sensitive no noise and 
insignificant factors, whereas RF handles high-dimensional data well, reduces overfitting,
it is more memory efficient, and can generally be more accurate, which better suits the
purposes of this fraud detection model.
"""
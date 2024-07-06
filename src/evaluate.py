import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Paths to model and data files
model_path = '../models/multinomialNB.pkl'
data_path = '../data/preprocessed_data.pkl'

# Load the model and data
model = joblib.load(model_path)
X_train, X_test, y_train, y_test = joblib.load(data_path)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Print results
print("Accuracy Score:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", pd.DataFrame(class_report).iloc[:-1, :].T)

# Save the metrics
metrics_path = '../output/'
os.makedirs(metrics_path, exist_ok=True)
joblib.dump(accuracy, f'{metrics_path}/accuracy_score.pkl')
joblib.dump(conf_matrix, f'{metrics_path}/confusion_matrix.pkl')
joblib.dump(class_report, f'{metrics_path}/classification_report.pkl')

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(f'{metrics_path}/confusion_matrix.png')
plt.show()

# Plot Accuracy Score
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='blue')
plt.ylim(0, 1)
plt.title('Accuracy Score')
plt.ylabel('Score')
plt.savefig(f'{metrics_path}/accuracy_score.png')
plt.show()

# Plot Predicted vs Actual Accuracy by Class
plt.figure(figsize=(6, 4))
accuracy_by_class = []
for label in np.unique(y_test):
    accuracy_by_class.append(accuracy_score(y_test[y_test == label], y_pred[y_test == label]))
plt.bar(np.unique(y_test), accuracy_by_class, color='green', alpha=0.6, label='Predicted')
actual_accuracy_by_class = [np.mean(y_test == label) for label in np.unique(y_test)]
plt.bar(np.unique(y_test), actual_accuracy_by_class, color='red', alpha=0.4, label='Actual')
plt.xlabel('Class')
plt.ylabel('Score')
plt.legend()
plt.title('Predicted vs Actual Accuracy by Class')
plt.savefig(f'{metrics_path}/accuracy_by_class.png')
plt.show()

# Plot Classification Report
class_report_df = pd.DataFrame(class_report).iloc[:-1, :].T
plt.figure(figsize=(12, 8))
sns.heatmap(class_report_df, annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report')
plt.savefig(f'{metrics_path}/classification_report.png')
plt.show()

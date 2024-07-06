from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np


# Load the preprocessed data
X_train, X_test, y_train, y_test = joblib.load('../data/preprocessed_data.pkl')

# Determine the unique classes in y_train
unique_classes = np.unique(y_train)

# Scale the data without centering
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = MultinomialNB()

# Fit the model incrementally
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, '../models/multinomialNB.pkl')

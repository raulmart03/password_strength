import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import os
import re
from sklearn.model_selection import train_test_split

# Define the correct path to the CSV file
file_path = os.path.join(os.path.dirname(__file__), '../data/data.csv')

# Function to clean the passwords
def clean_password(password):
    password = re.sub(r'\s+', '', password)
    return password

# Load the data, specifying the expected columns
try:
    data = pd.read_csv(file_path, usecols=['password', 'strength'])
except pd.errors.ParserError as e:
    print(f"Error parsing file: {e}")
    exit(1)

data.dropna(inplace=True)  # Remove rows with missing values

# Convert 'strength' to numeric, setting invalid values as NaN
data['strength'] = pd.to_numeric(data['strength'], errors='coerce')

# Filter passwords that are only in classes 0, 1, or 2
data = data[data['strength'].isin([0, 1, 2])]

# Check if there is data after filtering
if data.empty:
    print("No data available after filtering for classes 0, 1, and 2.")
    exit(1)

# Clean the passwords
data['password'] = data['password'].apply(clean_password)

if data['password'].str.len().sum() == 0:
    print("No valid passwords after cleaning.")
    exit(1)

print(f"Number of samples after cleaning: {len(data)}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
try:
    X = vectorizer.fit_transform(data['password'])
except ValueError as e:
    print(f"Error during vectorization: {e}")
    exit(1)

y = data['strength']

# Save the vectorizer
os.makedirs('../models', exist_ok=True)
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Save the preprocessed datasets
os.makedirs('../data', exist_ok=True)
joblib.dump((X_train, X_test, y_train, y_test), '../data/preprocessed_data.pkl')

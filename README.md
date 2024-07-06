#Password Strength Classifier

##Overview
The Password Strength Classifier project aims to predict the strength of passwords using machine learning. The project utilizes a Multinomial Naive Bayes model to classify passwords into three categories: weak, medium, and strong. The classifier is trained on a dataset of passwords labeled with their corresponding strength levels.

##Data Preprocessing
The raw data (data.csv) should be placed in the data/ directory.
 Run the data preprocessing script to clean the data, apply TF-IDF vectorization, and balance the classes using SMOTE:

python src/data_preprocessing.py

The preprocessed data will be saved as preprocessed_data.pkl in the data/ directory.

##Training the Model

Run the training script to train the Multinomial Naive Bayes model on the preprocessed data:

python src/training.py

The trained model and the TF-IDF vectorizer will be saved in the models/ directory.

##Evaluating the Model

Run the evaluation script to assess the performance of the trained model:

python src/evaluation.py

The evaluation metrics, including accuracy score, confusion matrix, and classification report, will be saved in the output/ directory. Visualizations of these metrics will also be saved as images.
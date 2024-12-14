Movie Rating Prediction using Machine Learning
This project aims to predict whether a movie will have a high or low rating based on various features such as genre, release year, and the number of ratings it has received. The dataset consists of movie details and user ratings, and several machine learning models (SVM, Random Forest, KNN) are used to perform the prediction.

Project Overview
The goal is to classify movies into two categories:

High Rating (1): Movies with an average rating of 4 or above.
Low Rating (0): Movies with an average rating below 4.
Three different machine learning algorithms are employed to make predictions on movie ratings:

Support Vector Machine (SVM) with an RBF kernel.
Random Forest Classifier.
K-Nearest Neighbors (KNN).
The models are evaluated using classification metrics, confusion matrices, and accuracy comparisons.

Datasets
movies.csv: Contains information about movies, including their unique movieId, title, and genre(s).
ratings.csv: Contains user ratings for movies, with movieId and the corresponding rating for each user.
Data Preprocessing Steps:
Merging of movies.csv and ratings.csv on movieId.
Creation of a binary target variable indicating high (>= 4) and low (< 4) ratings.
Extraction of the year from the movie title.
One-hot encoding of movie genres.
Aggregation of ratings by movie, including average ratings and count of ratings.
Normalization of features using StandardScaler.
Requirements
Python 3.x
pandas
numpy
scikit-learn
matplotlib
seaborn
You can install the required libraries using the following:

bash
Copy code
pip install -r requirements.txt
Where requirements.txt includes:

Copy code
pandas
numpy
scikit-learn
matplotlib
seaborn
Usage
Load and preprocess the data:

Load the datasets from movies.csv and ratings.csv.
Merge the data, preprocess it (extract year, one-hot encode genres, etc.), and prepare it for model training.
Train the models:

Train three different machine learning models: SVM, Random Forest, and KNN.
Evaluate the performance of the models using accuracy, confusion matrices, and classification reports.
Evaluate and visualize the results:

Print classification reports to compare the performance of the models.
Display confusion matrices for each model to visualize true vs predicted labels.
Plot a bar chart comparing the accuracy of the models.
Results
The performance of each model is evaluated on the test set, and the following evaluation metrics are provided:

Accuracy: The proportion of correct predictions.
Confusion Matrix: Shows the count of true positives, true negatives, false positives, and false negatives.
Classification Report: Includes precision, recall, and F1-score for each class.
Example Output
Classification Reports for each model:

plaintext
Copy code
SVM:
precision    recall  f1-score   support

      0       0.88      0.79      0.83       500
      1       0.80      0.89      0.84       500

accuracy                           0.84      1000
Confusion Matrices:

Plots showing how the predictions of each model compare with the actual values.
Model Accuracy Comparison: A bar plot comparing the accuracy of SVM, Random Forest, and KNN.

Future Work
Hyperparameter Tuning: Use Grid Search or Randomized Search to tune the models' hyperparameters for better performance.
Cross-validation: Implement cross-validation for more robust evaluation.
Additional Features: Consider adding other features, such as movie description or director information, to improve the model.
Deep Learning: Explore using deep learning models such as neural networks for movie rating prediction.
License
This project is licensed under the MIT License.

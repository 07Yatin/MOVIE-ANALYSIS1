import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Datasets
movies_path = 'movies.csv'  # Replace with your movies dataset path
ratings_path = 'ratings.csv'  # Replace with your ratings dataset path

# Load datasets
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Step 2: Data Preprocessing
# Merge datasets on movieId
data = pd.merge(ratings, movies, on='movieId')

# Create binary target variable: 1 for high (>=4), 0 for low (<4)
data['rating_class'] = (data['rating'] >= 4).astype(int)

# Extract year from title (if available)
data['year'] = data['title'].str.extract(r'\((\d{4})\)').astype(float)

# Split genres into individual columns (one-hot encoding)
genres = data['genres'].str.get_dummies(sep='|')

# Aggregate ratings by movie (mean rating and count of ratings)
movie_features = data.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    rating_count=('rating', 'count')
).reset_index()

# Merge aggregated features with genres and year
movie_data = pd.merge(movie_features, genres, left_on='movieId', right_index=True)
movie_data = pd.merge(movie_data, data[['movieId', 'year']].drop_duplicates(), on='movieId')

# Fill any missing years with median
movie_data['year'] = movie_data['year'].fillna(movie_data['year'].median())

# Features and Target
X = movie_data.drop(columns=['movieId', 'avg_rating', 'rating_count'])  # Features
y = (movie_data['avg_rating'] >= 4).astype(int)  # High/Low classification

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
# 1. Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 3. K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# Step 4: Evaluation
models = {'SVM': svm_pred, 'Random Forest': rf_pred, 'KNN': knn_pred}
accuracies = {}

print("Classification Reports:\n")
for model_name, predictions in models.items():
    print(f"{model_name}:")
    print(classification_report(y_test, predictions))
    accuracies[model_name] = accuracy_score(y_test, predictions)

# Plot Confusion Matrices
cm_labels = np.unique(y_test)  # Dynamic labels
plt.figure(figsize=(15, 5))
for i, (model_name, predictions) in enumerate(models.items(), 1):
    cm = confusion_matrix(y_test, predictions, labels=cm_labels)
    plt.subplot(1, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plt.tight_layout()
plt.show()

# Plot Accuracy Comparison
plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.ylim(0, 1)  # Set y-axis range from 0 to 1
plt.show()

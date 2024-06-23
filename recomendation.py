import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

# Sample movie ratings data (user-item matrix)
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'movie_id': [1, 2, 3, 2, 3, 1, 4, 5, 4, 5],
    'rating': [5, 3, 2, 4, 1, 5, 2, 3, 4, 1]
}

df = pd.DataFrame(data)

# Create user-item matrix (pivot table)
matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')

# Fill NaN values with 0 (means user hasn't rated that movie)
matrix = matrix.fillna(0)

# Convert pandas DataFrame to scipy sparse matrix
matrix_sparse = csc_matrix(matrix.values)

# Perform Singular Value Decomposition (SVD)
U, sigma, Vt = svds(matrix_sparse, k=2)  # k is the number of singular values to compute

sigma = np.diag(sigma)  # Construct diagonal array in SVD

# Predicted ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + np.mean(matrix.values, axis=1).reshape(-1, 1)

# Example: Recommend movies for a user
user_id = 1
user_ratings = predicted_ratings[user_id - 1]

# Get unrated movies for the user
rated_movies = df[df['user_id'] == user_id]['movie_id'].values
unrated_movies = [movie_id for movie_id in matrix.columns if movie_id not in rated_movies]

# Sort unrated movies based on predicted ratings
recommended_movies = sorted(zip(unrated_movies, user_ratings), key=lambda x: x[1], reverse=True)

print(f"Recommended movies for user {user_id}:")
for movie, rating in recommended_movies:
    print(f"Movie ID: {movie}, Predicted Rating: {rating:.2f}")

    
import numpy as np
from numpy import genfromtxt
from scipy.sparse.linalg import svds


# build a (i x u) rating matrix from the training csv file, where i = num. of items and u = num. of users
def rating_matrix(matrix):
    n_users, n_items = matrix[:, :2].astype(int).max(axis=0)
    rating_matrix = np.zeros((n_users, n_items), dtype=np.single)
    for user_id, item_id, rating, _ in matrix:
        rating_matrix[int(user_id) - 1, int(item_id) - 1] = rating
    return rating_matrix

train_data = genfromtxt('CW1/train_100k_withratings.csv', delimiter=',')
#train_data = genfromtxt('CW2/train_20M_withratings.csv', delimiter=',')
ratings = rating_matrix(train_data)

# Perform SVD
k = 50  # Number of singular values and vectors to compute
U, sigma, Vt = svds(ratings, k=k)

# Make predictions
predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)

print(ratings)
print(predicted_ratings)


print(ratings.shape)
print(predicted_ratings.shape)
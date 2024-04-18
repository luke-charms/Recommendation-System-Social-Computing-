import numpy as np
from scipy.sparse.linalg import svds
from numpy import genfromtxt


# training function for cosine similarity recommender system algorithm
def train_model(train_data = None):
    user_item_rating_matrix = rating_matrix(train_data)
    return user_item_rating_matrix

# build a (i x u) rating matrix from the training csv file, where i = num. of items and u = num. of users
def rating_matrix(matrix):
    n_users, n_items = matrix[:, :2].astype(int).max(axis=0)
    rating_matrix = np.zeros((n_users, n_items), dtype=int)
    for user_id, item_id, rating, _ in matrix:
        rating_matrix[int(user_id) - 1, int(item_id) - 1] = rating
    return rating_matrix


# switch test scores to 0 values to make hidden
def make_rating_test(ratingMatrix=None, test_values=None):
    copy_ratingMatrix = np.copy(ratingMatrix)
    for line in test_values:
        user = int(line[0])-1
        item = int(line[1])-1
        copy_ratingMatrix[user, item] = 0
    return copy_ratingMatrix


# rounds the final rating-matrix to nearest int
def round_rating(ratingMatrix=None):
    #final_ratingMatrix = np.zeros_like(ratingMatrix)
    for index, value in np.ndenumerate(ratingMatrix):
        ratingMatrix[index] = round(value)
    return ratingMatrix


class ItemPredictionSystem:
    def __init__(self, ratings_matrix, num_factors=10):
        self.ratings_matrix = ratings_matrix
        self.num_factors = num_factors
        self.user_factors = None
        self.item_factors = None
        self.mean_rating = None

    def train(self):
        # Center the ratings matrix by subtracting the mean rating of each user
        self.mean_rating = np.nanmean(self.ratings_matrix, axis=1)
        centered_ratings_matrix = self.ratings_matrix - self.mean_rating[:, np.newaxis]

        # Perform Singular Value Decomposition (SVD) to factorize the ratings matrix
        U, sigma, Vt = svds(centered_ratings_matrix, k=self.num_factors)

        # Reverse the diagonal matrix to obtain singular values in descending order
        sigma = np.diag(sigma[::-1])

        # Compute user and item factors
        self.user_factors = np.dot(U, np.sqrt(sigma))
        self.item_factors = np.dot(np.sqrt(sigma), Vt)


    def predict_rating(self, user_id, item_id):
        user_idx = user_id - 1  # Adjusting for 0-based indexing
        item_idx = item_id - 1  # Adjusting for 0-based indexing

        if user_idx < 0 or user_idx >= self.ratings_matrix.shape[0]:
            raise ValueError("Invalid user id")

        if item_idx < 0 or item_idx >= self.ratings_matrix.shape[1]:
            raise ValueError("Invalid item id")

        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained")

        # Predict the rating by taking dot product of user and item factors
        predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[:, item_idx])

        print("pred:", predicted_rating)
        
        # Add back the mean rating of the user
        predicted_rating += self.mean_rating[user_idx]
        
        return predicted_rating
    
    def final_matrix(self):
        predictions = np.dot(self.user_factors, self.item_factors)
        #predictions = np.add(predictions, self.mean_rating, axis=1)
        return predictions





train_data = genfromtxt('CW1/train_100k_withratings.csv', delimiter=',')
#train_data = genfromtxt('CW2/train_20M_withratings.csv', delimiter=',')
ratings = rating_matrix(train_data)


testing_indexes = np.random.randint(90570, size=18114)
test = train_data[testing_indexes,:]

rating_matrix_test = make_rating_test(ratingMatrix=ratings, test_values=test)
ratings_matrix = np.array_split(rating_matrix_test, 3, axis=1)


print("FINISHED MAKING RATING MATRIX!")

item_prediction_system = ItemPredictionSystem(rating_matrix_test, num_factors=10)
item_prediction_system.train()
item_prediction_system_predictions = np.dot(item_prediction_system.user_factors, item_prediction_system.item_factors).astype(np.single)
print(item_prediction_system_predictions.shape)
print("mean rating:", item_prediction_system.mean_rating.shape, "\n")


# Create and train the recommendation system
item_prediction_system1 = ItemPredictionSystem(ratings_matrix[0], num_factors=10)
item_prediction_system1.train()
item_prediction_system1_predictions = np.dot(item_prediction_system1.user_factors, item_prediction_system1.item_factors).astype(np.single)

print("FINSIHED TRAINING 1st PRED. SYS")
print(item_prediction_system1_predictions.shape, "\n")


# Create and train the recommendation system
item_prediction_system2 = ItemPredictionSystem(ratings_matrix[1], num_factors=10)
item_prediction_system2.train()
item_prediction_system2_predictions = np.dot(item_prediction_system2.user_factors, item_prediction_system2.item_factors).astype(np.single)

print("FINSIHED TRAINING 2nd PRED. SYS")
print(item_prediction_system2_predictions.shape, "\n")


# Create and train the recommendation system
item_prediction_system3 = ItemPredictionSystem(ratings_matrix[2], num_factors=10)
item_prediction_system3.train()
item_prediction_system3_predictions = np.dot(item_prediction_system3.user_factors, item_prediction_system3.item_factors).astype(np.single)

print("FINSIHED TRAINING 3rd PRED. SYS")
print(item_prediction_system3_predictions.shape, "\n")


print("------------------")

predictions = np.concatenate((item_prediction_system1_predictions, item_prediction_system2_predictions, item_prediction_system3_predictions), axis=1)
meanRatings = np.stack((item_prediction_system1.mean_rating, item_prediction_system2.mean_rating, item_prediction_system3.mean_rating), axis=1)
meanRatings = np.nanmean(meanRatings, axis=1)
print("PRED. SYS")
print(predictions.shape)
print(meanRatings.shape, "\n")

#print(item_prediction_system_predictions, "\n\n", predictions)

final = item_prediction_system.final_matrix()
#final_rating_item = round_rating(ratingMatrix=final)
#final_rating_item123 = round_rating(ratingMatrix=predictions)

print("train rating \n", rating_matrix_test, "\n and final shape:", rating_matrix_test.shape)
print(final)

#print("final rating \n", final_rating_item, "\n and final shape:", final_rating_item.shape)
#print("final rating 123 \n", final_rating_item123, "\n and final 123 shape:", final_rating_item123.shape)


# Predict rating for a specific user and item
user_id = 1
item_id = 1
predicted_rating = item_prediction_system.predict_rating(user_id, item_id)
print(f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating}")

pred_rating = predictions[0,0]
print("pred2:", pred_rating)
pred_rating += meanRatings[0]
print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating}")

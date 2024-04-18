import numpy as np
import logging

# setup logging parameters
LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

### %========== MATRIX FUNCTIONS ==========%
     
# build a (i x u) rating matrix from the training csv file, where i = num. of items and u = num. of users
def rating_matrix(matrix):
    n_users, n_items = matrix[:, :2].astype(int).max(axis=0)
    rating_matrix = np.zeros((n_users, n_items), dtype=float)
    for user_id, item_id, rating, _ in matrix:
        rating_matrix[int(user_id) - 1, int(item_id) - 1] = rating
    logger.info( 'Finished building rating matrix' )
    return rating_matrix

# function that performs matrix factorisation
def matrix_factorization(R, K=3, steps=5000, alpha=0.0002, beta=0.02):
    '''
    R: rating matrix
    P: User features matrix ( |U| * K )
    Q: Item features matrix ( K * |I| )
    K: latent features
    steps: iterations (epochs)
    alpha: learning rate
    beta: regularisation parameter
    '''

    # setup the User features and Item features matrices
    P = np.random.rand(len(R), K)
    Q = np.random.rand(len(R[0]), K)
    Q = Q.T

    # iterate for 'steps' epochs
    for step in range(steps):
        print("Epoch", step, "starting to train")
        # iterates over the rating matrix
        for index, value in np.ndenumerate(R):
            # checks to see if the rating matrix index has a rating or not (0 represents no rating present)
            if value > 0:
                # calculate predicted rating of the matrix factorisation matrix by performing the dot product between User features and Item features matrices
                rating_prediction = np.dot(P[index[0],:],Q[:,index[1]])
                # calculate the error between the real value (from the rating matrix) and the predicted value
                error = np.subtract(value, rating_prediction)

                # for each value in the User features and Item features matrices, adjust the value based on the error found:
                #
                #       q_i = q_i + γ * (error * p_u - λ * q_i)
                #       p_u = p_u + γ * (error * p_u - λ * p_u)
	            #   where
                #     q_i = user vectors
	            #     p_u = item vectors
	            #     γ = learning rate
                #     λ = regularisation parameter
                #     error = r_ui - q_i^T.p_u (error calculated above)
	            #
                for k in range(K):
                        P[index[0]][k] += np.multiply(alpha, (np.subtract(np.multiply(error, Q[k][index[1]]), np.multiply(beta, P[index[0]][k]))))
                        Q[k][index[1]] += np.multiply(alpha, (np.subtract(np.multiply(error, P[index[0]][k]), np.multiply(beta, Q[k][index[1]]))))
    
    return P, Q.T



### %========== ADMIN FUNCTIONS ==========%

# switch test scores to 0 values to make hidden
def make_rating_test(ratingMatrix=None, test_values=None):
    copy_ratingMatrix = np.copy(ratingMatrix)
    for line in test_values:
        # adjust for 0 indexing
        user = int(line[0])-1
        item = int(line[1])-1
        copy_ratingMatrix[user, item] = 0
    return copy_ratingMatrix



# rounds the final rating-matrix to nearest 0.5
def round_rating(ratingMatrix=None):
    for index, value in np.ndenumerate(ratingMatrix):
        ratingMatrix[index] = (round(value * 2) / 2)
    return ratingMatrix
            

# serialize a set of predictions to file
def serialize_predictions(output_file = None, ratingMatrix = None, test_values = None):
    for line in test_values:
        print(line)
        user = int(line[0])-1
        item = int(line[1])-1
        score = ratingMatrix[user, item]
        line[2] = score
    np.savetxt(output_file, test_values, '%s', delimiter=",")


# training function for matrix factorisation recommender system algorithm
def train_model(train_data = None):
    logger.info( '%=== Beginning to train model ===%' )
    logger.info( 'Making Rating Matrix...' )
    user_item_rating_matrix = rating_matrix(train_data)
    return user_item_rating_matrix


# load a set of ratings from file
def load_data(file = None, train=True):
    with open(file) as f:
          file_lines = f.readlines()
    matrix = np.asarray([line.strip('\n').split(',') for line in file_lines])
    if not train:
         matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1), dtype=matrix.dtype)))
         matrix[:, [2, 3]] = matrix[:, [3, 2]]
    return matrix


# main function to execute code
def main():
    logger.info( 'Training prediction functions on the train data...' )
    logger.info( 'Generating array from .csv file' )

    train_data = load_data( file = 'CW2/train_20M_withratings.csv' )
    test = load_data( file = 'CW2/test_20M_withoutratings.csv', train=False)

    rating_matrix_train = train_model(train_data=train_data)
    print(rating_matrix_train)

    ITER_STEPS = 50
    K = 20
    
    nP, nQ = matrix_factorization(rating_matrix_train, K=K, steps=ITER_STEPS)

    nR = np.dot(nP, nQ.T)

    final_rating = round_rating(ratingMatrix=nR)
    print("train rating \n", rating_matrix_train, "\n and final shape:", rating_matrix_train.shape)
    print("final rating \n", final_rating, "\n and final shape:", final_rating.shape)
	
    serialize_predictions(output_file='CW2/submission.csv', ratingMatrix=rating_matrix_train, test_values=test)
    logger.info( 'Predictions saved to file submission.csv' )
    


if __name__ == '__main__':
    main()
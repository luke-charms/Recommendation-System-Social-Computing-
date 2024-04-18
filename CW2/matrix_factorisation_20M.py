import numpy as np
from numpy import genfromtxt
import sys
import logging
import time

from sklearn.metrics import mean_absolute_error

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

### %========== MATRIX FUNCTIONS ==========%
     
# build a (i x u) rating matrix from the training csv file, where i = num. of items and u = num. of users
def rating_matrix(matrix):
    n_users, n_items = matrix[:, :2].astype(int).max(axis=0)
    rating_matrix = np.zeros((n_users, n_items), dtype=float)
    #rating_matrix[:] = np.nan
    for user_id, item_id, rating, _ in matrix:
        rating_matrix[int(user_id) - 1, int(item_id) - 1] = rating
    logger.info( 'Finished building rating matrix' )
    return rating_matrix


def matrix_factorization(R, indices, K=3, steps=5000, alpha=0.0002, beta=0.02):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter
    '''
    P = np.random.rand(len(R), K)
    Q = np.random.rand(len(R[0]), K)
    Q = Q.T

    for step in range(steps):
        print("Epoch", step, "starting to train")
        #print("P:", P)
        #print("Q:", Q.T)
        for index, value in np.ndenumerate(R):
            if value > 0:
                #print(index)
                # calculate error
                eij = np.subtract(value, np.dot(P[index[0],:],Q[:,index[1]]))

                for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[index[0]][k] += np.multiply(alpha, (np.subtract(np.multiply(eij, Q[k][index[1]]), np.multiply(beta, P[index[0]][k]))))
                        Q[k][index[1]] += np.multiply(alpha, (np.subtract(np.multiply(eij, P[index[0]][k]), np.multiply(beta, Q[k][index[1]]))))
                        
                        #P[index[0]][k] += np.multiply(alpha, (np.subtract(np.multiply(np.multiply(2, eij), Q[k][index[1]]), np.multiply(beta, P[index[0]][k]))))
                        #Q[k][index[1]] += np.multiply(alpha, (np.subtract(np.multiply(np.multiply(2, eij), P[index[0]][k]), np.multiply(beta, Q[k][index[1]]))))
    
    return P, Q.T

def get_indices(test_values = None):
    indices = []
    for line in test_values:
        user = int(line[0])-1
        item = int(line[1])-1
        indices.append((user,item))
    return indices


### %========== ADMIN FUNCTIONS ==========%

# switch test scores to 0 values to make hidden
def make_rating_test(ratingMatrix=None, test_values=None):
    copy_ratingMatrix = np.copy(ratingMatrix)
    for line in test_values:
        user = int(line[0])-1
        item = int(line[1])-1
        copy_ratingMatrix[user, item] = 0
    return copy_ratingMatrix

# calculate rmse between real and predicted scores
def compare_score(ratingMatrix=None, test_values=None):
    real_scores = []
    predicted_scores = []
    for line in test_values:
        real_score = float(line[2])
        predicted_score = ratingMatrix[int(line[0])-1, int(line[1])-1]
        real_scores.append(real_score)
        predicted_scores.append(predicted_score)
    real_scores = np.array(real_scores)
    predicted_scores = np.array(predicted_scores)
    print("Real scores:", real_scores)
    print("Predicted scores:", predicted_scores)
    rme = mean_absolute_error(real_scores, predicted_scores)
    return rme


# rounds the final rating-matrix to nearest int
def round_rating(ratingMatrix=None):
    #final_ratingMatrix = np.zeros_like(ratingMatrix)
    for index, value in np.ndenumerate(ratingMatrix):
        ratingMatrix[index] = round(value)
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


# training function for cosine similarity recommender system algorithm
def train_model(train_data = None):
    logger.info( '%=== Beginning to train model ===%' )
    logger.info( 'Making Rating Matrix...' )
    user_item_rating_matrix = rating_matrix(train_data)
    #print(user_item_rating_matrix, "\n")

    #user_item_rating_matrix_mean = np.mean(user_item_rating_matrix, axis = 1)
    #user_item_rating_matrix_demeaned = user_item_rating_matrix - user_item_rating_matrix_mean.reshape(-1, 1)
    #print(user_item_rating_matrix_demeaned)
	
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

    train_data_20M = genfromtxt('CW2/train_20M_withratings.csv', delimiter=',')
    #train_data_20M = load_data( file = 'CW2/train_20M_withratings.csv' )
    #test_data_20M = load_data( file = 'CW2/train_20M_withoutratings.csv', train=False)

    testing_indexes_20M = np.random.randint(18615333, size=3723066)
    test_20M = train_data_20M[testing_indexes_20M,:]

    rating_matrix_train = train_model(train_data=train_data_20M)
    #rating_matrix_test = make_rating_test(ratingMatrix=rating_matrix_train, test_values=test_20M)

    indices = get_indices(test_20M)

    print(rating_matrix_train.shape)

    rating_matrices = np.array_split(rating_matrix_train, 3, axis=1)

    print(rating_matrices[0].shape)
    print(rating_matrices[1].shape)
    print(rating_matrices[2].shape)

    ITER_STEPS = 20
    K = 3
    
    currentTime = time.time()

    nP1, nQ1 = matrix_factorization(rating_matrices[0], indices, K=K, steps=ITER_STEPS)
    #nP2, nQ2 = matrix_factorization(rating_matrices[1], indices, K=K, steps=ITER_STEPS)
    #nP3, nQ3 = matrix_factorization(rating_matrices[2], indices, K=K, steps=ITER_STEPS)

    stopTime = time.time()
    timetaken = stopTime - currentTime
    print("Algorithm took: ", round(timetaken,1), "s to run")

    nR1 = np.dot(nP1, nQ1.T)
    #nR2 = np.dot(nP2, nQ2.T)
    #nR3 = np.dot(nP3, nQ3.T)

    print(nR1.shape)
    #print(nR2.shape)
    #print(nR3.shape)

    np.savetxt("CW2/nR1.csv", nR1, delimiter=",")
    #np.savetxt("CW2/nR2.csv", nR2, delimiter=",")
    #np.savetxt("CW2/nR3.csv", nR3, delimiter=",")

    """nR1_LOAD = genfromtxt('CW2/nR1.csv', delimiter=',')
    nR2_LOAD = genfromtxt('CW2/nR2.csv', delimiter=',')
    nR3_LOAD = genfromtxt('CW2/nR3.csv', delimiter=',')

    nR_half = np.concatenate((nR1_LOAD, nR2_LOAD), axis=1)
    nR = np.concatenate((nR_half, nR3_LOAD), axis=1)

    nR_half_TEST = np.concatenate((nR1, nR2), axis=1)
    nR_TEST = np.concatenate((nR_half, nR3), axis=1)

    print("ARRAYS EQUAL? : ", np.array_equal(nR_TEST, nR))

    final_rating = round_rating(ratingMatrix=nR)
    print("train rating \n", rating_matrix_test, "\n and final shape:", rating_matrix_test.shape)
    print("final rating \n", final_rating, "\n and final shape:", final_rating.shape)

    rmse = compare_score(final_rating, test)
    print("RMSE:", rmse)
	
    #serialize_predictions(output_file='CW2/submission.csv', ratingMatrix=rating_matrix_test, test_values=test)
    #logger.info( 'Predictions saved to file submission.csv' )"""
    


if __name__ == '__main__':
    main()
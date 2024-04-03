import numpy as np
import logging

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

### %========== MATRIX FUNCTIONS ==========%
     
# build a (u x i) rating matrix from the training csv file, where u = num. of users and i = num. of items
def rating_matrix(matrix):
    n_users, n_items = matrix[:, :2].astype(int).max(axis=0)
    rating_matrix = np.empty((n_users, n_items), dtype=np.float16)
    rating_matrix[:] = np.nan
    for user_id, item_id, rating, _ in matrix:
        rating_matrix[int(user_id) - 1, int(item_id) - 1] = rating
    logger.info( 'Finished building rating matrix' )
    return rating_matrix


# apply mean centering across the axis' of the rating matrix
def mean_centered_rating_matrix(rating_matrix):
    def mean_rating(u_rating):
        mean = np.nanmean(u_rating[u_rating.nonzero()])
        return mean

    means = np.apply_along_axis(mean_rating, 1, rating_matrix).reshape((rating_matrix.shape[0], 1))
    indx = np.where(rating_matrix == np.nan)
    rating_matrix = rating_matrix - means
    rating_matrix[indx] = np.nan
    return rating_matrix


# build a (u x u) user similarity matrix from the rating matrix, where u = num. of users
def user_similarity_matrix(rating_matrix):
    counter = -1
    n_users = rating_matrix.shape[0]
    similarity_matrix = np.empty((n_users, n_users))
    similarity_matrix[:] = np.nan
    print("Number of users: ", n_users)
    for index, _ in np.ndenumerate(similarity_matrix):
        if np.isnan(similarity_matrix[index]) and index[0] != index[1]:
            #sim = cosine_similarity(rating_matrix[u1], rating_matrix[u2])
            sim = pearson_similarity(rating_matrix, index[0]+1, index[1]+1)
            similarity_matrix[index[0], index[1]] = sim
            similarity_matrix[index[1], index[0]] = sim
        if counter != index[0]:
            print( 'User ', index[0]+1, '\'s /', n_users, 'matrix has been made' )
            counter += 1
    return similarity_matrix


# build a (i x i) item similarity matrix from the rating matrix, where i = num. of items
def item_similarity_matrix(rating_matrix):
    counter = -1
    n_items = rating_matrix.shape[1]
    similarity_matrix = np.empty((n_items, n_items))
    similarity_matrix[:] = np.nan
    print("Number of items: ", n_items)
    for index, _ in np.ndenumerate(similarity_matrix):
        if np.isnan(similarity_matrix[index]) and index[0] != index[1]:
            sim = adjusted_cosine_similarity(rating_matrix, index[0]+1, index[1]+1)
            similarity_matrix[index[0], index[1]] = sim
            similarity_matrix[index[1], index[0]] = sim
        if counter != index[0]:
            print( 'Item ', index[0]+1, '\'s /', n_items, 'matrix has been made' )
            counter += 1
    return similarity_matrix


### %========== SIMILARITY FUNCTIONS ==========%

# calculate the cosine similarity between two users
def cosine_similarity(x, y):
    # find the dot product between the vectors
    dot_product = np.dot(x, y)
    
    # find the magnitudes of each vector
    magnitude_x = np.sqrt(np.sum(x**2)) 
    magnitude_y = np.sqrt(np.sum(y**2))
    
    #       cos(Θ) = A•B / ||A|| * ||B||
	#   where
    #     cos(Θ) = cosine similarity angle
	#     A = matrix1
	#     B = matrix2
	#
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)
    return cosine_similarity


# calculate the pearson similarity between two users
def pearson_similarity(rating_matrix, userA=None, userB=None):
    # get the two arrays of user scores whose similarity will be calculated
    u1_array = rating_matrix[userA-1].astype(np.double)
    u2_array = rating_matrix[userB-1].astype(np.double)

    # find the average user score for the two users whose similarity will be calculated
    # (this makes it easier when subtracting the average score from each known score later on)
    u1_avg_score = np.nanmean(u1_array)
    u2_avg_score = np.nanmean(u2_array)

    # find all the indexes of the items where both users have a registered score, by removing any columns where either user has a NaN (non-input) score
    u1_nonNaN_indices = np.argwhere(~np.isnan(u1_array))
    u2_nonNaN_indices = np.argwhere(~np.isnan(u2_array))
    nonNaN_indexes = np.intersect1d(u1_nonNaN_indices, u2_nonNaN_indices)

	# compute the difference between the score user 1 gave each rated item and their average score
	# the first bracket of the numerator function for pearson similarity
    u1_top = u1_array[nonNaN_indexes] - u1_avg_score

    # compute the difference between the score user 2 gave each rated item and their average score
	# the second bracket of the numerator function for pearson similarity
    u2_top = u2_array[nonNaN_indexes] - u2_avg_score

    # compute the square of the difference between the score user 1 gave each rated item and their average score
	# the first bracket of the denominator function for pearson similarity
    u1_sqr = np.square(u1_array[nonNaN_indexes] - u1_avg_score)

    # compute the square of the difference between the score user 2 gave each rated item and their average score
	# the second bracket of the denominator function for pearson similarity
    u2_sqr = np.square(u2_array[nonNaN_indexes] - u2_avg_score)

    #       ∑i=I(ru1,i - ^ru1)*(ru2,i - ^ru2) / √∑i=I(ru1,i - ^ru1)^2 * √∑i=I(ru2,i - ^ru2)^2
	#   where
    #     i = neighbourhood of users
	#     r1 = user1 rating
	#     r2 = user2 rating
    #     ^ru1 = user1 average rating
    #     ^ru2 = user2 average rating
	#
    pearson_similarity = np.sum(np.multiply((u1_top), u2_top)) / (np.sqrt(np.sum(u1_sqr)) * np.sqrt(np.sum(u2_sqr)))
    return pearson_similarity


# calculate the cosine similarity between two items
def adjusted_cosine_similarity(rating_matrix, item1, item2):
    # get the two arrays of item scores whose similarity will be calculated
    i1_array = rating_matrix[:, item1-1].astype(np.double)
    i2_array = rating_matrix[:, item2-1].astype(np.double)

    # find the average user score for each user of the rating matrix, needed to calculate the cosine similarity
    # (this makes it easier when subtracting the average score from each known score later on)
    user_average_matrix = np.nanmean(rating_matrix, axis=1)

    # find all the indexes of the items where both items have a registered score, by removing any columns where either item has a NaN (non-input) score
    i1_nonNaN_indices = np.argwhere(~np.isnan(i1_array))
    i2_nonNaN_indices = np.argwhere(~np.isnan(i2_array))
    nonNaN_indexes = np.intersect1d(i1_nonNaN_indices, i2_nonNaN_indices)

    # compute the difference between the score useri gave the rated item1 and their average score
	# the first bracket of the numerator function for adjusted cosine similarity
    i1_top = i1_array[nonNaN_indexes] - user_average_matrix[nonNaN_indexes]

    # compute the difference between the score useri gave the rated item2 and their average score
	# the second bracket of the numerator function for adjusted cosine similarity
    i2_top = i2_array[nonNaN_indexes] - user_average_matrix[nonNaN_indexes]

    # compute the square of the difference between the score useri gave the rated item1 and their average score
	# the first bracket of the denominator function for pearson similarity
    i1_sqr = np.square(i1_array[nonNaN_indexes] - user_average_matrix[nonNaN_indexes])

    # compute the square of the difference between the score useri gave the rated item2 and their average score
	# the second bracket of the denominator function for pearson similarity
    i2_sqr = np.square(i2_array[nonNaN_indexes] - user_average_matrix[nonNaN_indexes])

    denominator = (np.multiply(np.sqrt(np.sum(i1_sqr)), np.sqrt(np.sum(i2_sqr))))
    if denominator == 0:
        return 0
    #       ∑u=U(ru,i2 - ^ru)*(ru,i2 - ^ru) / √∑u=U(ru,i1 - ^ru)^2 * √∑u=U(ru,i2 - ^ru)^2
	#   where
    #     u = neighbourhood of users
	#     ru,i1 = item1 rating
	#     ru,i2 = item2 rating
    #     ^ru = average user rating for that user
	#
    cosine_similarity = np.divide(np.sum(np.multiply((i1_top), i2_top)), denominator)
    return cosine_similarity


### %========== NEIGHBOURHOOD FUNCTIONS ==========%

# returns a neighbourhood of users that have the closest similarity to the target user
def get_user_neighbourhood(user_similarity_matrix, rating_matrix, user=None, item=None, neigh_num=0):
    user_similarities = user_similarity_matrix[user-1]
    available_users = {}

    for index, val in np.ndenumerate(user_similarities):
        user_ratings = rating_matrix[index[0]]
        if not np.isnan(val) and not np.isnan(user_ratings[item-1]):
            available_users[val] = index[0]

    users = [abs(x) for x in available_users.keys()]
    users_sort = sorted(users, reverse=True)
    highest_predictions = users_sort[:neigh_num]

    neighbourhood = []
    for i in range(0, len(highest_predictions)):
        user_neigh = highest_predictions[i]
        if user_neigh > 0.3:
        #if True:
            if user_neigh in available_users.keys():
                neighbourhood.append(available_users[user_neigh] + 1)
            else:
                user_neigh = user_neigh*-1
                neighbourhood.append(available_users[user_neigh] + 1)
    return neighbourhood


# returns a neighbourhood of items that have the closest similarity to the target item
def get_item_neighbourhood(item_similarity_matrix, rating_matrix, user=None, item=None, neigh_num=0):
    item_similarities = item_similarity_matrix[item-1]
    available_items = {}

    for index, val in np.ndenumerate(item_similarities):
        item_ratings = rating_matrix[index[0]]
        if not np.isnan(val) and not np.isnan(item_ratings[user-1]):
            available_items[val] = index[0]
    
    items = [abs(x) for x in available_items.keys()]
    items_sort = sorted(items, reverse=True)
    highest_predictions = items_sort[:neigh_num]

    neighbourhood = []
    for i in range(0, len(highest_predictions)):
         item_neigh = highest_predictions[i]
         if item_neigh in available_items.keys():
             neighbourhood.append(available_items[item_neigh] + 1)
         else:
              item_neigh = item_neigh*-1
              neighbourhood.append(available_items[item_neigh] + 1)
    return neighbourhood


### %========== PREDICTION FUNCTIONS ==========%

# predict an user rating based off the data in the similarity and rating matrix, using the calculated neighbourhood
def predict_rating_user_based(user_similarity_matrix, rating_matrix, user=None, item=None, neighbourhood=[]):
    if len(neighbourhood) == 0:
        return 2.5
    user_average_matrix = np.nanmean(rating_matrix, axis=1)
    neighbourhood_adjusted = np.subtract(neighbourhood,1)
    similarity = user_similarity_matrix[user-1][neighbourhood_adjusted]

    top_fraction = np.sum(np.multiply(similarity, (np.subtract(rating_matrix[:, item-1][neighbourhood_adjusted], user_average_matrix[neighbourhood_adjusted]))))
    bot_fraction = np.sum(np.abs(similarity))
    
    user_pred = user_average_matrix[user-1] + np.divide(top_fraction, bot_fraction)
    return user_pred


# predict an item rating based off the data in the similarity and rating matrix, using the calculated neighbourhood
def predict_rating_item_based(item_similarity_matrix, rating_matrix, user=None, item=None, neighbourhood=[]):
    if len(neighbourhood) == 0:
        return 2.5
    user_average_matrix = np.nanmean(rating_matrix, axis=1)
    neighbourhood_adjusted = np.subtract(neighbourhood,1)
    similarity = item_similarity_matrix[item-1][neighbourhood_adjusted]

    top_fraction = np.sum(np.multiply(similarity, (np.subtract(rating_matrix[user-1][neighbourhood_adjusted], user_average_matrix[neighbourhood_adjusted]))))
    bot_fraction = np.sum(np.abs(similarity))
    
    item_pred = np.divide(top_fraction, bot_fraction)
    return item_pred


### %========== ADMIN FUNCTIONS ==========%


# serialize a set of predictions to file
def serialize_predictions(output_file = None, prediction_matrix = None ):
    np.savetxt(output_file, prediction_matrix, '%s', delimiter=",")


# training function for cosine similarity recommender system algorithm
def train_model( train_data = None, user_sim=False, item_sim=False,):
    logger.info( '%=== Beginning to train models ===%' )
    logger.info( 'Making Rating Matrix...' )
    user_item_rating_matrix = rating_matrix(train_data)
    print(user_item_rating_matrix)

    user_similarity_matrix_train = []
    item_similarity_matrix_train = []

    if user_sim:
        logger.info( 'Making User Similarity Matrix...' )
        user_similarity_matrix_train = user_similarity_matrix(rating_matrix=user_item_rating_matrix)
        logger.info( 'User Similarity Matrix MADE' )
        print(user_similarity_matrix_train)
    if item_sim:
        logger.info( 'Making Item Similarity Matrix...' )
        item_similarity_matrix_train = item_similarity_matrix(rating_matrix=user_item_rating_matrix)
        logger.info( 'Item Similarity Matrix MADE' )
        print(item_similarity_matrix_train)
	
    return user_item_rating_matrix, item_similarity_matrix_train, user_similarity_matrix_train


# infer function for cosine similarity recommender system algorithm
def predict_ratings( test_data = None, item_sim_matrix = None, user_sim_matrix = None, rating_matrix = None, type=None, verbose=False, neigh_num=0):
    testing_data = np.copy(test_data)
    # TODO: need absolute values or not for prediction scores?
    if type == 'user':
        return predict_ratings_user(test_data=testing_data, user_sim_matrix=user_sim_matrix, rating_matrix=rating_matrix, verbose=verbose, neigh_num=neigh_num)
    elif type == 'item':
        return predict_ratings_item(test_data=testing_data, item_sim_matrix=item_sim_matrix, rating_matrix=rating_matrix, verbose=verbose)
    else:
        return predict_ratings_average(test_data=testing_data, item_sim_matrix=item_sim_matrix, user_sim_matrix=user_sim_matrix, rating_matrix=rating_matrix, verbose=verbose)


# find the predicted scores for each value in the test set, using either a user-based or item-based approach, or both
def predict_ratings_user( test_data = None, user_sim_matrix = None, rating_matrix = None, verbose=False, neigh_num=0):
    for i in range(0,len(test_data[:])): 
      user_pred = int(test_data[i][0])
      item_pred = int(test_data[i][1])

      user_neighbourhood = get_user_neighbourhood(user_sim_matrix, rating_matrix, user=user_pred, item=item_pred, neigh_num=neigh_num)
      prediction_from_user = abs( predict_rating_user_based(user_sim_matrix, rating_matrix, user=user_pred, item=item_pred, neighbourhood=user_neighbourhood))
      
      if verbose:
        print("\n%=== PREDICTION FOR user:", user_pred, " and item:", item_pred, " ===%")
        print("User neighbourhood: ", user_neighbourhood)
        print("Item predicted score (from user based): ", float(round(prediction_from_user * 2) / 2))
  
      test_data[i][2] = float(round(prediction_from_user))
      #test_data[i][2] = float(round(prediction_from_user * 2) / 2)
    
    return test_data


# find the predicted scores for each value in the test set using a user-based approach
def predict_ratings_item( test_data = None, item_sim_matrix = None, rating_matrix = None, verbose=False):
    for i in range(0,len(test_data[:])): 
      user_pred = int(test_data[i][0])
      item_pred = int(test_data[i][1])

      item_neighbourhood = get_item_neighbourhood(item_sim_matrix, rating_matrix, user=user_pred, item=item_pred)  
      prediction_from_item = abs( predict_rating_item_based(item_sim_matrix, rating_matrix, user=user_pred, item=item_pred, neighbourhood=item_neighbourhood))

      if verbose:
        print("\n%=== PREDICTION FOR user: ", user_pred, " and item: ", item_pred, " ===%")
        print("Item neighbourhood: ", item_neighbourhood)
        print("Item predicted score (from item based): ", round(prediction_from_item,0))

      test_data[i][2] = float(round(prediction_from_item))
    
    return test_data


# find the predicted scores for each value in the test set using a mixture of user and item based approach, averaging the answer
def predict_ratings_average( test_data = None, item_sim_matrix = None, user_sim_matrix = None, rating_matrix = None, verbose=False):
    for i in range(0,len(test_data[:])): 
      user_pred = int(test_data[i][0])
      item_pred = int(test_data[i][1])

      item_neighbourhood = get_item_neighbourhood(item_sim_matrix, rating_matrix, user=user_pred, item=item_pred)
      user_neighbourhood = get_user_neighbourhood(user_sim_matrix, rating_matrix, user=user_pred, item=item_pred)
      
      prediction_from_item = abs( predict_rating_item_based(item_sim_matrix, rating_matrix, user=user_pred, item=item_pred, neighbourhood=item_neighbourhood))
      prediction_from_user = abs( predict_rating_user_based(user_sim_matrix, rating_matrix, user=user_pred, item=item_pred, neighbourhood=user_neighbourhood))
  
      rounded_score = round((prediction_from_item + prediction_from_user) / 2)

      if verbose:
        print("\n%=== PREDICTION FOR user: ", user_pred, " and item: ", item_pred, " ===%")
        print("Item neighbourhood: ", item_neighbourhood, " and user neighbourhood: ", user_neighbourhood)
        print("Item predicted score (from item based): ", prediction_from_item, " and (from user based): ", prediction_from_user)
        print("Rounded scores: ", round(prediction_from_item,0), " and ", round(prediction_from_user,0))
        print("Average between two scores: ", rounded_score)

      test_data[i][2] = float(rounded_score)
    
    return test_data
      

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

    num_neighbours = 90

    train_data = load_data( file = 'CW1/train_100k_withratings.csv' )
    test = load_data( file = 'CW1/test_100k_withoutratings.csv', train=False)

    #train_data = load_data(file='train_10_withratings.csv', train=True)
    #test = load_data(file='test_10_withoutratings.csv', train=False)

    #testing_indexes = np.random.randint(90570, size=18114)
    #training = np.arange(0, 90570, 1, dtype=int)
    #training_indexes = [x for x in training if x not in testing_indexes]

    #train = train_data[training_indexes,:]
    #test = train_data[testing_indexes,:]

    rating_matrix_train, item_sim_matrix, user_sim_matrix = train_model(train_data=train_data,  user_sim=True, item_sim=False)
    
    pred_user = predict_ratings(test_data=test, 
                                item_sim_matrix=item_sim_matrix, 
                                user_sim_matrix=user_sim_matrix, 
                                rating_matrix=rating_matrix_train, 
                                type='user', verbose=True, neigh_num=num_neighbours)

    #pred_item = predict_ratings(test_data=test, 
    #                            item_sim_matrix=item_sim_matrix, 
    #                            user_sim_matrix=user_sim_matrix, 
    #                            rating_matrix=rating_matrix_train, 
    #                            type='item', verbose=False, neigh_num=num_neighbours)
    #pred_avg = predict_ratings(test_data=test, 
    #                           item_sim_matrix=item_sim_matrix, 
    #                           user_sim_matrix=user_sim_matrix, 
    #                           rating_matrix=rating_matrix_train, 
    #                           type='average', verbose=False, neigh_num=num_neighbours)

    """
    pred_user_20 = predict_ratings(test_data=test, 
                                item_sim_matrix=item_sim_matrix, 
                                user_sim_matrix=user_sim_matrix, 
                                rating_matrix=rating_matrix_train, 
                                type='user', verbose=True, neigh_num=20)
    pred_user_40 = predict_ratings(test_data=test, 
                                item_sim_matrix=item_sim_matrix, 
                                user_sim_matrix=user_sim_matrix, 
                                rating_matrix=rating_matrix_train, 
                                type='user', verbose=True, neigh_num=40)
    pred_user_60 = predict_ratings(test_data=test, 
                                item_sim_matrix=item_sim_matrix, 
                                user_sim_matrix=user_sim_matrix, 
                                rating_matrix=rating_matrix_train, 
                                type='user', verbose=True, neigh_num=60)
    pred_user_80 = predict_ratings(test_data=test, 
                                item_sim_matrix=item_sim_matrix, 
                                user_sim_matrix=user_sim_matrix, 
                                rating_matrix=rating_matrix_train, 
                                type='user', verbose=True, neigh_num=80)
    pred_user_100 = predict_ratings(test_data=test, 
                                item_sim_matrix=item_sim_matrix, 
                                user_sim_matrix=user_sim_matrix, 
                                rating_matrix=rating_matrix_train, 
                                type='user', verbose=True, neigh_num=100)
    pred_user_120 = predict_ratings(test_data=test, 
                                item_sim_matrix=item_sim_matrix, 
                                user_sim_matrix=user_sim_matrix, 
                                rating_matrix=rating_matrix_train, 
                                type='user', verbose=True, neigh_num=120)
    """

    serialize_predictions(output_file='CW1/submission.csv', prediction_matrix=pred_user)
    logger.info( 'Predictions saved to file submission.csv' )


    y_true = np.array(test[:, 2], dtype=np.float64)
    y_pred_user = np.array(pred_user[:, 2], dtype=np.float64)

    #y_pred_item = np.array(pred_item[:, 2], dtype=np.float64)
    #y_pred_avg = np.array(pred_avg[:, 2], dtype=np.float64)
    
    #y_pred_user_20 = np.array(pred_user_20[:, 2], dtype=np.float64)
    #y_pred_user_40 = np.array(pred_user_40[:, 2], dtype=np.float64)
    #y_pred_user_60 = np.array(pred_user_60[:, 2], dtype=np.float64)
    #y_pred_user_80 = np.array(pred_user_80[:, 2], dtype=np.float64)
    #y_pred_user_100 = np.array(pred_user_100[:, 2], dtype=np.float64)
    #y_pred_user_120 = np.array(pred_user_120[:, 2], dtype=np.float64)


    print("REAL:", "\n",
        y_true, "\n",
        "USER :", "\n",
        y_pred_user, "\n",)
    #    "USER (20):", "\n",
    #    y_pred_user_20, "\n",)
    #    "USER (40):", "\n",
    #    y_pred_user_40, "\n",
    #    "USER (60):", "\n",
    #    y_pred_user_60, "\n",
    #    "USER (80):", "\n",
    #    y_pred_user_80, "\n",
    #    "USER (100):", "\n",
    #    y_pred_user_100, "\n",
    #    "USER (120):", "\n",
    #    y_pred_user_120, "\n",)
    #    "ITEM:", "\n",
    #    y_pred_item,  "\n",)
    #    "AVERAGE", "\n",
    #    y_pred_avg)

    from sklearn.metrics import mean_absolute_error
    y_pred_user_MAE = mean_absolute_error(y_true, y_pred_user)
    #y_pred_items_MAE = mean_absolute_error(y_true, y_pred_item)
    #y_pred_avg_both_MAE = mean_absolute_error(y_true, y_pred_avg)
    
    #y_pred_user_MAE_20 = mean_absolute_error(y_true, y_pred_user_20)
    #y_pred_user_MAE_40 = mean_absolute_error(y_true, y_pred_user_40)
    #y_pred_user_MAE_60 = mean_absolute_error(y_true, y_pred_user_60)
    #y_pred_user_MAE_80 = mean_absolute_error(y_true, y_pred_user_80)
    #y_pred_user_MAE_100 = mean_absolute_error(y_true, y_pred_user_100)
    #y_pred_user_MAE_120 = mean_absolute_error(y_true, y_pred_user_120)


    print(
          "%== MAE Scores ==%", "\n",
          "User based: ", y_pred_user_MAE,"\n",)
    #      "User based (20): ", y_pred_user_MAE_20,"\n",
    #      "User based (40): ", y_pred_user_MAE_40,"\n",
    #      "User based (60): ", y_pred_user_MAE_60,"\n",
    #      "User based (80): ", y_pred_user_MAE_80,"\n",
    #      "User based (100): ", y_pred_user_MAE_100,"\n",
    #      "User based (120): ", y_pred_user_MAE_120,"\n",)
    #      "Item based: ", y_pred_items_MAE,)
    #      "Average of both: ", y_pred_items_MAE,
    #      """)



if __name__ == '__main__':
    main()
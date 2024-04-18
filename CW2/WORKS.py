from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split

from surprise import SVD
from surprise import SVDpp
from surprise import NMF

from surprise import accuracy
from surprise.model_selection import GridSearchCV
import numpy as np
from numpy import genfromtxt

import pandas as pd

# Load your data (replace this with your dataset)
reader = Reader(line_format='user item rating', sep=',')
#data = Dataset.load_from_file('CW1/train_100k_withratings.csv', reader=reader)
data = Dataset.load_from_file('CW2/train_20M_withratings.csv', reader=reader)

#test = genfromtxt('CW1/test_100k_withoutratings.csv', delimiter=',', dtype='U')
test = genfromtxt('CW2/test_20M_withoutratings.csv', delimiter=',', dtype='U')


# Split the dataset into training and testing sets
#trainset, testset = train_test_split(data, test_size=0.1)

"""param_grid = {
    'n_factors': [10, 50, 100, 200, 500], 
    'n_epochs':[10, 20, 30, 50], 
    'lr_all':[0.005, 0.001, 0.02], 
    'reg_all':[0.02, 0.1, 0.4]}


gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3, joblib_verbose=True)

gs.fit(data)

# best RMSE score
print(gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print(gs.best_params["rmse"])

# We can now use the algorithm that yields the best rmse:
algo = gs.best_estimator["rmse"]
algo.fit(data.build_full_trainset())"""


# Use the SVD algorithm for matrix factorization
algo = SVD(n_factors=200, n_epochs=30, lr_all=0.02, reg_all=0.1, verbose=True)
#algo = SVDpp(n_factors=20, n_epochs=30, lr_all=0.02, reg_all=0.1, verbose=True)
#algo = NMF(n_factors=50, n_epochs=100, reg_pu=0.04,  verbose=True)
algo.fit(data.build_full_trainset())
#algo.fit(trainset)

# Make predictions on the test set
#predictions = algo.test(testset)

# Evaluate the model's accuracy
#accuracy.rmse(predictions)

#trainset.to_raw_uid(1), trainset.to_raw_iid(60)

# serialize a set of predictions to file
def serialize_predictions(output_file = None, algorithm=None, test_values=None):
    for line in test_values:
        user = str(int(line[0]))
        item = str(int(line[1]))
        score = algo.predict(user,item)
        score = float(round(score.est * 2) / 2)
        #print("-----")
        #print(line)
        #print(score)
        line[2] = score
    np.savetxt(output_file, test_values, '%s', delimiter=",")

test = np.insert(test, 2, 0, axis=1)
serialize_predictions(output_file='CW2/submission.csv', algorithm=algo, test_values=test)

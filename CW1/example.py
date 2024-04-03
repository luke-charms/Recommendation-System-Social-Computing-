
"""
df = pd.read_csv('test_10_withratings.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]

print(train)
print("")
print(test)
"""
import pandas as pd
from scipy.spatial.distance import pdist, squareform

M = np.asarray([[8,3,4,2,7],
             [2,3,5,7,5],
             [5,4,7,4,7],
             [7,1,7,3,8],
             [1,7,4,6,5],
             [8,3,8,3,7]])

M_u = M.mean(axis=1)
item_mean_subtracted = M - M_u[:, None]
similarity_matrix = 1 - squareform(pdist(item_mean_subtracted.T, 'cosine'))
print(similarity_matrix)

n = len(M[0]) # find out number of columns(items)
normalized = item_mean_subtracted/np.linalg.norm(item_mean_subtracted, axis = 0).reshape(1,n) #divide each column by its norm, normalize it
normalized = normalized.T #transpose it
similarity_matrix2 = np.asarray([[np.inner(normalized[i],normalized[j] ) for i in range(n)] for j in range(n)]) # compute the similarity matrix by taking inner product of any two items
print(similarity_matrix2)



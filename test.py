import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer

import sklearn

ct = ColumnTransformer(
    [("norm1", Normalizer(norm='l1'), [0, 1]),
     ("norm2", Normalizer(norm='l1'), [2, 3])])

normaliser_transformer = ColumnTransformer(transformers=[("norm", sklearn.preprocessing.StandardScaler(), [0, 1])])

transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [0, 1])], remainder='passthrough')
X = np.array([[0., 1., 2., 2.],
              [1., 1., 0., 1.]])
# Normalizer scales each row of X to unit norm. A separate scaling
# is applied for the two first and two last elements of each
# row independently.
print(normaliser_transformer.fit(X))


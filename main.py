import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing

df = pd.read_csv ('prueba.csv', delimiter=' ')
df = df.values
X, y = df[:,:129], df[:,-1]

# result = []

# for row in y:
#     if np.array_equal(np.array(row), np.array([1, 0, 0, 0, 0])):
#         result.append(1)            
#     elif np.array_equal(np.array(row), np.array([0, 1, 0, 0, 0])):
#         result.append(2)            
#     elif np.array_equal(np.array(row), np.array([0, 0, 1, 0, 0])):
#         result.append(3)            
#     elif np.array_equal(np.array(row), np.array([0, 0, 0, 1, 0])):
#         result.append(4)            
#     elif np.array_equal(np.array(row), np.array([0, 0, 0, 0, 1])):
#         result.append(5)            
#     elif np.array_equal(np.array(row), np.array([0, 0, 0, 0, 0])):
#         result.append(6)         
#     else:
#         result.append(7)         

# y = np.array(result)
# print(y)
oversample = RandomUnderSampler()
X, y = oversample.fit_sample(X, y)

y = y.astype('int')
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(y)+1))
y = label_binarizer.transform(y)

# r = []
# for row in y:
#     if row == 1:
#         r.append(np.array([1, 0, 0, 0, 0, 0]))                   
#     elif row == 2:
#         r.append(np.array([0, 1, 0, 0, 0, 0]))                
#     elif row == 3:
#         r.append(np.array([0, 0, 1, 0, 0, 0]))                   
#     elif row == 4:
#         r.append(np.array([0, 0, 0, 1, 0, 0]))               
#     elif row == 5:
#         r.append(np.array([0, 0, 0, 0, 1, 0]))                   
#     elif row == 6:
#         r.append(np.array([0, 0, 0, 0, 0, 1]))    
#     else:
#         r.append(np.array([0, 0, 0, 0, 0, 0]))    

# print(np.array(r).shape)
result = np.concatenate((X, y), axis=1)
print(y.shape)
np.random.shuffle(result)
np.savetxt('p2.csv', result, delimiter=';', fmt='%.0f')
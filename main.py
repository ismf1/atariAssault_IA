import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing

df = pd.read_csv ('dataBuena.csv', delimiter=';')
df = df.values
X, y = df[:,:59], df[:,59:]

print(sorted(Counter(y).items()))

result    = []
y_reverse = []
i = 0

for row in y:
    a = False
    index = 0
    for idx, x in enumerate(result):
        if np.array_equal(x, row):
            a = True
            index = idx
    if a:
        y_reverse.append(y_reverse[index])
    else:
        y_reverse.append(i)
        result.append(row)
        i += 1

y = np.array(y_reverse)

oversample = RandomOverSampler()
X, y = oversample.fit_sample(X, y)

print(sorted(Counter(y).items()))

# y = y.astype('int')
# label_binarizer = sklearn.preprocessing.LabelBinarizer()
# label_binarizer.fit(range(max(y)+1))
# y = label_binarizer.transform(y)

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
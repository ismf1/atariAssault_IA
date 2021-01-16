import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing

df = pd.read_csv('../../neuralNetwork/dataTrain.txt', delimiter=';')
df = df.sample(frac=1).reset_index(drop=True)

target_count = df.iloc[:, 59:]

print(target_count.iloc[:, 0].value_counts())
print(target_count.iloc[:, 1].value_counts())
print(target_count.iloc[:, 2].value_counts())
print(target_count.iloc[:, 3].value_counts())
print(target_count.iloc[:, 4].value_counts())

a = []
for i in range(0, 64):
    a.append(str(i))

df.columns = a
df_max = df[(df['59'] == 0) & 
         (df['60'] == 0) & 
         (df['61'] == 0) & 
         (df['62'] == 0) & 
         (df['63'] == 0)][:40000]
df_min = df[(df['59'] == 1) | 
         (df['60'] == 1) | 
         (df['61'] == 1) | 
         (df['62'] == 1) | 
         (df['63'] == 1)]

df = pd.concat([df_max, df_min])
df = df.sample(frac=1).reset_index(drop=True)

# X_train = df.iloc[:, :59]
# y_train = df.iloc[:, 59:]

# scaler = preprocessing.StandardScaler().fit(X_train)
# x_scaled = scaler.transform(X_train)

# df = np.concatenate([x_scaled, y_train], axis=1)

np.savetxt("data.txt", df, delimiter=";", fmt='%.8f')




# df_max = df[(df['59'] == 0) & 
#          (df['60'] == 0) & 
#          (df['61'] == 0) & 
#          (df['62'] == 0) & 
#          (df['63'] == 0)]

# print(len(df_max))

# df_minority_upsampled  = resample(df_min, 
#                                  replace=True,    # sample without replacement
#                                  n_samples=19102,     # to match minority class
#                                  random_state=123)

# pd.concat([df_max, df_minority_upsampled]).sample(frac=1).reset_index(drop=True).to_csv('prueba2.csv', index=False)

# target_count.plot(kind='bar', title='Count (target)')

import pandas as pd
import numpy as np
from sklearn.utils import resample

df = pd.read_csv('./ic.csv', delimiter=';')

target_count = df.iloc[:, 59:]

print(target_count.iloc[:, 0].value_counts())
print(target_count.iloc[:, 1].value_counts())
print(target_count.iloc[:, 2].value_counts())
print(target_count.iloc[:, 3].value_counts())
print(target_count.iloc[:, 4].value_counts())


df.sample(frac=1).reset_index(drop=True).to_csv('data2.csv', index=False)

# a = []
# for i in range(0, 64):
#     a.append(str(i))

# df.columns = a

# df_max = df[(df['59'] == 0) & 
#          (df['60'] == 0) & 
#          (df['61'] == 0) & 
#          (df['62'] == 0) & 
#          (df['63'] == 0)]
# df_min = df[(df['59'] == 1) | 
#          (df['60'] == 1) | 
#          (df['61'] == 1) | 
#          (df['62'] == 1) | 
#          (df['63'] == 1)]

# print(len(df_max))

# df_minority_upsampled  = resample(df_min, 
#                                  replace=True,    # sample without replacement
#                                  n_samples=19102,     # to match minority class
#                                  random_state=123)

# pd.concat([df_max, df_minority_upsampled]).sample(frac=1).reset_index(drop=True).to_csv('prueba2.csv', index=False)

# target_count.plot(kind='bar', title='Count (target)')
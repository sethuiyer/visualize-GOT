'''Question is to which outcome is the NaN in attacker_outcome of last row of the dataset is most similar to.

'''
#Step 1: Import dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#Step 2: Load the dataset
battles = pd.read_csv("dataset/battles.csv")
y=battles["attacker_outcome"].fillna("NaN")

#Step 3: Count the number of NaN in each column
counter_nan = battles.isnull().sum()
counter_without_nan=counter_nan[counter_nan == 0]

#Step 4: Clean the Dataset
battles = battles[counter_without_nan.keys()]


#Step 5: Create feature vectors

#Encode the categorical variables
le=LabelEncoder()
for col in battles.columns.values:
    if battles[col].dtypes == "object":
        unique_labels=np.unique(battles[col])
        le.fit(unique_labels)
        battles[col]=le.transform(battles[col])

#Encode the class label data
le.fit(np.unique(y))
y=le.transform(y)

#Standardize the feature vectors
X = battles.ix[:].values
standard_scalar = StandardScaler()
x_std = standard_scalar.fit_transform(X)

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization

tsne = TSNE(n_components=2, random_state=0)
x_std_2d = tsne.fit_transform(x_std)

markers=('s', 'd', 'o')
color_map = {0:'red', 1:'blue', 2:'lightgreen'}

plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=x_std_2d[y==cl,0], y=x_std_2d[y==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()




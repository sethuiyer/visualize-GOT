'''Question is how allegiances, nobility and the appearence in the book affect the gender of charecter deaths.

'''
#Step 1: Import dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

#Step 2: Load the dataset
chardeaths = pd.read_csv("dataset/character-deaths.csv")
y=chardeaths["Gender"]
chardeaths=chardeaths.drop(chardeaths.columns[[0]],1)
counter_nan = chardeaths.isnull().sum()
counter_without_nan=counter_nan[counter_nan == 0]
chardeaths = chardeaths[counter_without_nan.keys()]
chardeaths=chardeaths.drop("Gender",axis=1)

le = LabelEncoder()
le.fit(np.unique(y))
y_encoded=le.transform(y)
le.fit(np.unique(chardeaths["Allegiances"]))
chardeaths["Allegiances_encoded"]=le.transform(chardeaths["Allegiances"])
chardeaths=chardeaths.drop("Allegiances",axis=1)
#chardeaths=chardeaths.drop(chardeaths.columns[[1,2,3,4,5]],axis=1)
# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
print chardeaths.columns.values


#Standardize the feature vectors
X = chardeaths.ix[:].values
standard_scalar = StandardScaler()
x_std = standard_scalar.fit_transform(X)

test_percentage = 0.1
x_train, x_test, y_train, y_test = train_test_split(x_std, y_encoded, test_size = test_percentage, random_state = 0)


tsne = TSNE(n_components=2, random_state=0)
x_std_2d = tsne.fit_transform(x_test)



markers=('o', 'o')
color_map = {0:'red', 1:'blue'}

plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=x_std_2d[y_test==cl,0], y=x_std_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()




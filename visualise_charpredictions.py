''' 0 is not popular and not died , 1 is not popular and died , 2 is popular and not died 
    3 is popular and died '''
''' Will t-sne visualization identify the close similarity between 0 & 1 and 3 &4 or 0 & 2 and 1 & 3 ?'''

#Step 1: Import dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

#Step 2: Load the dataset
death_preds = pd.read_csv("dataset/character-predictions.csv")
death_preds.loc[:, "title"] = pd.factorize(death_preds.title)[0]
death_preds.loc[:, "culture"] = pd.factorize(death_preds.culture)[0]
death_preds.loc[:, "mother"] = pd.factorize(death_preds.mother)[0]
death_preds.loc[:, "father"] = pd.factorize(death_preds.father)[0]
death_preds.loc[:, "heir"] = pd.factorize(death_preds.heir)[0]
death_preds.loc[:, "house"] = pd.factorize(death_preds.house)[0]
death_preds.loc[:, "spouse"] = pd.factorize(death_preds.spouse)[0]

death_preds.drop(["S.No","name", "pred", "plod", "isAlive","culture","popularity"], 1, inplace = True)
death_preds.columns = map(lambda x: x.replace(".", "").replace("_", ""), death_preds.columns)
death_preds.fillna(value = -1, inplace = True)
y=death_preds["isPopular"]

print death_preds.columns.values


#Standardize the feature vectors

le= LabelEncoder()
le.fit(np.unique(y))
y_encoded=le.transform(y)

le.fit(np.unique(death_preds["actual"]))
y2 = le.transform(death_preds["actual"])
death_preds.drop(["actual"],1,inplace = True)

X = death_preds.ix[:].values
standard_scalar = StandardScaler()
x_std = standard_scalar.fit_transform(X)

Y=2*y2+y
le.fit(np.unique(Y))
Y=le.transform(Y)


test_percentage = 0.5
x_train, x_test, y_train, y_test = train_test_split(x_std, Y, test_size = test_percentage, random_state = 0)


tsne = TSNE(n_components=2, random_state=0)
x_std_2d = tsne.fit_transform(x_test)



markers=('s', 'd', 'o', '^', 'v')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
plt.figure()
for idx, cl in enumerate(np.unique(Y)):
    plt.scatter(x=x_std_2d[y_test==cl,0], y=x_std_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()



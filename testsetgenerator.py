#This script contains different method to divide a
#given dataset into training set and testing set into given ratio
#and also gives procedure to visualise the given data
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit

def load_house_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_path=os.path.join(dir_path, "housing.csv")
    return pd.read_csv(csv_path)

#def tsc(i,tr,hash):
#    return hash(np.int64(i)).digest()[-1]<256*tr

#def split_data(data, test_ratio, id, hash=hashlib.md5):
#    ids=data[id]
#    its=ids.apply(lambda id_:tsc(id_,test_ratio,hash))
#    return data.loc[~its], data.loc[its]

housing=load_house_data()
housing.info()
#print(housing["ocean_proximity"].value_counts())
#print(housing.describe())
#housing.hist(bins=50, figsize=(20,15))
#plt.show()
#housingwithid=housing.reset_index()
#housingwithid["id"]=housing["longitude"]*1000+housing["latitude"]
#train_set, test_set=split_data(housingwithid, 0.2, "index")
housing["income_cat"]=np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace=True)
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    train_set=housing.loc[train_index]
    test_set=housing.loc[test_index]

print(housing["income_cat"].value_counts()/len(housing))
for set_ in (train_set, test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing=train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()
corr_matrix=housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

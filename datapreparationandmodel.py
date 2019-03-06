import os
import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as rmse
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import GridSearchCV as gsc

class ComAttAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedperroom=True):
        self.add_bedperroom=add_bedperroom

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        roomsperhousehold=X[:,rooms_ix]/X[:,households_ix]
        popperhouse=X[:,population_ix]/X[:,households_ix]
        if self.add_bedperroom:
            bedperroom=X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, roomsperhousehold, popperhouse, bedperroom]
        else:
            return np.c_[X, roomsperhousehold, popperhouse]

def load_house_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_path=os.path.join(dir_path, "housing.csv")
    return pd.read_csv(csv_path)

housing=load_house_data()
housing["income_cat"]=np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace=True)
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    train_set=housing.loc[train_index]
    test_set=housing.loc[test_index]

for set_ in (train_set, test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing=train_set.copy()
housing["roomsperhousehold"]=housing["total_rooms"]/housing["households"]
housing["bedperroom"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["popperhouse"]=housing["population"]/housing["households"]
housing=train_set.drop("median_house_value",axis=1)
housing_labels=train_set["median_house_value"].copy()
#median=housing["total_bedrooms"].median()
#housing["total_bedrooms"].fillna(median, inplace=True)
i=SimpleImputer(strategy="median")
housing_num=housing.drop("ocean_proximity", axis=1)
i.fit(housing_num)
print(i.statistics_)
print(housing_num.median().values)
X=i.transform(housing_num)
housing_tr=pd.DataFrame(X, columns=housing_num.columns)
oe=OrdinalEncoder()
housing_cat=housing[["ocean_proximity"]]
print(housing_cat.head(10))
housing_enc=oe.fit_transform(housing_cat)
print( housing_enc[:10])
ce=OneHotEncoder()
housing_1hot=ce.fit_transform(housing_cat)
print(ce.categories_)
rooms_ix, bedrooms_ix, population_ix, households_ix=3, 4, 5, 6
attradder=ComAttAdder(add_bedperroom=False)
housing_extraattrib=attradder.transform(housing.values)
numpipe=Pipeline([('i', SimpleImputer(strategy="median")),('attradder', ComAttAdder()),('sscal', StandardScaler())])
housing_num_tr=numpipe.fit_transform(housing_num)
nattr=list(housing_num)
cattr=["ocean_proximity"]
fp=ColumnTransformer([("num", numpipe, nattr),("cat", OneHotEncoder(), cattr)])
housing_final=fp.fit_transform(housing)
lr=LinearRegression()
lr.fit(housing_final, housing_labels)
sd=housing.iloc[:5]
sl=housing_labels.iloc[:5]
sdp=fp.transform(sd)
print("Predictions:",lr.predict(sdp))
print("Labels:",list(sl))
housing_predictions=lr.predict(housing_final)
le=rmse(housing_labels, housing_predictions)
lre=np.sqrt(le)
print(lre)
tr=DecisionTreeRegressor()
tr.fit(housing_final, housing_labels)
housing_predictions=tr.predict(housing_final)
tmse=rmse(housing_labels, housing_predictions)
trmse=np.sqrt(tmse)
print(trmse)
scores=cvs(tr, housing_final, housing_labels, scoring="neg_mean_squared_error", cv=10)
trmses=np.sqrt(-scores)
print("DecisionTreeRegressor")
print("Scores:",trmses)
print("Mean:",trmses.mean())
print("Standard Deviation:", trmses.std())
lscores=cvs(lr, housing_final, housing_labels, scoring="neg_mean_squared_error", cv=10)
lrmses=np.sqrt(-lscores)
print("LinearRegression")
print("Scores:",lrmses)
print("Mean:",lrmses.mean())
print("Standard Deviation:", lrmses.std())
fr=rfr()
fr.fit(housing_final, housing_labels)
housing_predictions=fr.predict(housing_final)
fmse=rmse(housing_labels, housing_predictions)
frmse=np.sqrt(fmse)
print(frmse)
fscores=cvs(fr, housing_final, housing_labels, scoring="neg_mean_squared_error", cv=10)
frmses=np.sqrt(-fscores)
print("RandomForestRegressor")
print("Scores:",frmses)
print("Mean:",frmses.mean())
print("Standard Deviation:", frmses.std())
pgrid=[{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},{'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}]
gs=gsc(fr, pgrid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
gs.fit(housing_final, housing_labels)
print(gs.best_params_)
print(gs.best_estimator_)
cres=gs.cv_results_
for ms, ps in zip(cres["mean_test_score"],cres["params"]):
    print(np.sqrt(-ms), ps)

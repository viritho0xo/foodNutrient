import sklearn
import pandas as pd
from numpy import mean 
from numpy import std
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from scipy import stats

# from keras.utils.np_utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# from keras.optimizers import RMSprop
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ReduceLROnPlateau
# load data from FoodNutrients directory to a numpy array
nutrients = open("nutrient.txt", "r").readlines()
nutrients = [x.strip() for x in nutrients]
def todf(filename):
    data = open("FoodNutrients/"+filename, "r").readlines()
    data = [list(map(float, x.split())) for x in data]
    data = pd.DataFrame(data, columns=nutrients)
    return data

X = todf("rawFoodWetNutri.txt")
# y = todf("FoodNutrients/cookedFoodWetNutri.txt")["Magnesium, Mg"]

# split data into train and test sets
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

# regression me
# regressor = RandomForestRegressor(n_estimators=100)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# graph me
# import matplotlib.pyplot as plt
# plt.scatter(y_test, y_pred)
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.show()

# give me all the graph possible
# import seaborn as sns
# sns.pairplot(X)
# plt.show()

# print("MSE: ", mean_squared_error(y_test, y_pred))
# print("MAE: ", mean_absolute_error(y_test, y_pred))
# print("R2: ", r2_score(y_test, y_pred))

def trainModel(X, y, label):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    RMSE = MSE ** 0.5
    SPC = stats.spearmanr(y_test, y_pred)[0]
    return [label, MSE, MAE, R2, RMSE, SPC]

def toSpreadSheet(raw, cooked, name):
    data = []
    for label in nutrients:
        data.append(trainModel(todf(raw+ ".txt"), todf(cooked + ".txt")[label], label))
    data = pd.DataFrame(data, columns=["Nutrient", "MSE", "MAE", "R2", "RMSE", "SPC"])
    data.to_csv("Evaluations/"+ name + "RandomForest.csv", index=False)
    
# toSpreadSheet("rawFoodWetNutri")
toSpreadSheet("rawFoodDryNutri", "cookedFoodDryNutri", "dryFood")
toSpreadSheet("rawFoodWetNutri", "cookedFoodWetNutri", "wetFood")

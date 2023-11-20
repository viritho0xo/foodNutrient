import sklearn
import pandas as pd
from numpy import mean 
from numpy import std
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor

# load data from FoodNutrients directory to a numpy array
nutrients = open("nutrient.txt", "r").readlines()
nutrients = [x.strip() for x in nutrients]
def todf(filename):
    data = open(filename, "r").readlines()
    data = [list(map(float, x.split())) for x in data]
    data = pd.DataFrame(data, columns=nutrients)
    return data

X = todf("FoodNutrients/rawFoodWetNutri.txt")
y = todf("FoodNutrients/cookedFoodWetNutri.txt")["Magnesium, Mg"]

# split data into train and test sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

# regression me
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# graph me
# import matplotlib.pyplot as plt
# plt.scatter(y_test, y_pred)
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.show()
# evaluate me
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
print("MSE: ", mean_squared_error(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))








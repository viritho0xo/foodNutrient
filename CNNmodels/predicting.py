import os
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Specify the directory path where the models are stored
models_directory = 'CNNmodels'
import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

# Get a list of all model files in the directory
model_files = [file for file in os.listdir(models_directory) if file.endswith('.h5')]

# Load each model and use it for prediction
for model_file in model_files[:1]:
    model_path = os.path.join(models_directory, model_file)
    model = load_model(model_path)
    
    # select the columns you want to predict
    # Perform prediction using the loaded model
    # Add your prediction code here
    # reshape the data
    
test_data = pd.read_csv("CNNpredictions/wetFoodRawFull.csv")
scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)
test_data = test_data.reshape((-1, 9, 3, 1))

def modelPredct(modifier, nutrient):
    model_file = modifier + "Food" + nutrient + ".h5"
    model_path = os.path.join(models_directory, model_file)
    model = load_model(model_path)
    pred = model.predict(test_data)
    return pred

predictNutrients = open("FoodNutrients/predictNutrient.txt", "r").readlines()
predictNutrients = [x.strip() for x in predictNutrients]


preds = pd.DataFrame()
for nutrient in predictNutrients:
    pred = modelPredct("wet", nutrient)
    pred = pd.DataFrame(pred, columns=[nutrient])
    preds = pd.concat([preds, pd.DataFrame(pred)], axis=1)
preds.to_csv("CNNpredictions/Predictions/wetFoodPreds1"+".csv", index=False)
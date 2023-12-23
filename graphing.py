import random
import statistics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import os

def generate_similar_list(original_list):
    similar_list = []
    std_dev = statistics.stdev(original_list) if len(original_list) > 1 else 0.1
    for value in original_list:
        similar_value = value + random.uniform(-std_dev*0.1, std_dev*0.1)
        similar_list.append(similar_value)
    return similar_list

# open csv file to pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv('Evaluations/dryFoodCNN.csv')
predictNutrients = open("FoodNutrients/predictNutrient.txt", "r").readlines()
predictNutrients = [x.strip() for x in predictNutrients]
baselineWeight = pd.read_csv("FoodNutrients/PredictNutri.csv")
nutrients = open("FoodNutrients/nutrient.txt", "r").readlines()
nutrients = [x.strip() for x in nutrients]
# for label in predictNutrients:
#     weight = baselineWeight.at
#     for val in baseline[label]:
#         baseline[label] = baseline[label].replace(val, val*(weight/100))
# baseline.to_csv("CNNpredictions/wetFoodBaseline.csv", index=False)
    
def camGen(cam, filename):
    plt.imshow(cam, "hot")
    plt.savefig('Graphs/featureRanking' + filename + '.png')
    plt.clf()
# read all the csv files in CNNimportance
# for filename in os.listdir('CNNimportance'):
#     cam = pd.read_csv('CNNimportance/' + filename)
#     camGen(cam, filename)

# for filename in os.listdir('CNNpredictions'):
dryCat = open("FoodCats/DryCat.txt", "r").readlines()
dryCat = [int(x.strip()) for x in dryCat]
dryU = [13, 5, 17, 12, 16]
wetCat = open("FoodCats/WetCat.txt", "r").readlines()
wetCat = [int(x.strip()) for x in wetCat]
wetU = [11, 24, 9, 20, 16, 12]

def baselineGen(filename, cat_list):
    raw = pd.read_csv("CNNpredictions/" + filename)
    for label in predictNutrients:
        i = 0
        for val in raw[label]:
            cat = cat_list[i]
            weight = float(baselineWeight.loc[baselineWeight["Cat"]==cat][label].iloc[0])
            raw[label] = raw[label].replace(val, val*(weight/100))
            i += 1
    raw.to_csv("CNNpredictions/"+ "wetFoodBaseline" +".csv", index=False)

def stats(true, pred):
    # calculate mse, mae, r2, rmse, spc
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mse)
    spc = 1 - (np.sqrt(mse) / np.mean(true))
    return mse, mae, r2, rmse, spc
    
def statCal(truename, predname, name, feat, cat_list):
    true = pd.read_csv("CNNpredictions/" + truename) 
    pred = pd.read_csv("CNNpredictions/Predictions/" + predname)
    corr = pd.DataFrame(columns=["Nutrient", "MSE", "MAE", "R2", "RMSE", "SPC"])
    trueStat = pd.DataFrame(columns=["Nutrient", "Avg", "StdDev"])
    predStat = pd.DataFrame(columns=["Nutrient", "Avg", "StdDev"])
    i = 0
    for label in predictNutrients:
        mse, mae, r2, rmse, spc = stats(true[label], pred[label])
        meanTrue = np.mean(true[label])
        meanPred = np.mean(pred[label])
        trueStdDev = statistics.stdev(true[label]) if len(true[label]) > 1 else 0.1
        predStdDev = statistics.stdev(pred[label]) if len(pred[label]) > 1 else 0.1
        # trueStat = pd.concat([trueStat, pd.DataFrame([[label, meanTrue, trueStdDev]], columns=["Nutrient", "Avg", "StdDev"])])
        # predStat = pd.concat([predStat, pd.DataFrame([[label, meanPred, predStdDev]], columns=["Nutrient", "Avg", "StdDev"])])
    # trueStat.to_csv("Graphs/Dist/" + name +"TrueDist.csv", index=False)
    # predStat.to_csv("Graphs/Dist/" + name + feat +"Dist.csv", index=False)
        corr = pd.concat([corr, pd.DataFrame([[label, mse, mae, r2, rmse, spc]], columns=["Nutrient", "MSE", "MAE", "R2", "RMSE", "SPC"])])
    # corr.to_csv("Graphs/Corr/" + name +"Stats.csv", index=False)
def statCal(true, pred):
    corr = pd.DataFrame(columns=["Nutrient", "MSE", "MAE", "R2", "RMSE", "SPC"])
    for label in predictNutrients:
        mse, mae, r2, rmse, spc = stats(true[label], pred[label])
        corr = pd.concat([corr, pd.DataFrame([[label, mse, mae, r2, rmse, spc]], columns=["Nutrient", "MSE", "MAE", "R2", "RMSE", "SPC"])])
    return corr

def addCat(raw, cat_list):
    raw = pd.concat([raw, pd.DataFrame(cat_list, columns=["Cat"])], axis=1)
    return raw

def rmCat(raw):
    raw = raw.drop(columns=["Cat"])
    return raw

predsDir = "CNNpredictions/Predictions/"

def catStats(dir, prefix):
    predictions = [filename for filename in os.listdir(dir) if filename.startswith(prefix)]
    if prefix == "dry":
        cat_list = dryCat
        unique = dryU
        true = pd.read_csv("CNNpredictions/dryFoodTrue.csv")
    else:
        cat_list = wetCat
        unique = wetU
        true = pd.read_csv("CNNpredictions/wetFoodTrue.csv")
    true = addCat(true, cat_list)        
    for file_name in predictions:
        if file_name.endswith(".csv"):
            pred = pd.read_csv(predsDir + file_name)
            pred = addCat(pred, cat_list)
            for val in unique:
                new_pred = pred[pred.Cat == val]
                new_true = true[true.Cat == val]
                new_true = rmCat(new_true)
                new_pred = rmCat(new_pred)
                catCorr = statCal(new_true, new_pred)
                catName = matchCat(val)
                catCorr.to_csv("Graphs/Corr/Cats/" + str(val) + "_" + prefix + catName + "Stats.csv", index=False)

def matchCat(val):
    match val:
        case 9:
            return "Fruits"
        case 11:
            return "Vegetables"
        case 12:
            return "Nut and Seed"
        case 16: 
            return "Legumes"
        case 20:
            return "Cereal Grains and Pasta"
        case 24:
            return "American Indian-Alaska Native Foods"
        case 5:
            return "Poultry"
        case 13:
            return "Beef"
        case 17:
            return "Game"
        case _:
            return
        
catStats(predsDir, "dry")
def getVal(filename, label):
    df = pd.read_csv("CNNpredictions/" + filename)
    df = df[label]
    return df




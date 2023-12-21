
import pandas as pd
from numpy import mean 
from numpy import std
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from scipy import stats
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
# load data from FoodNutrients directory to a numpy array

nutrients = open("FoodNutrients/nutrient.txt", "r").readlines()
nutrients = [x.strip() for x in nutrients]
predictNutrients = open("FoodNutrients/predictNutrient.txt", "r").readlines()
predictNutrients = [x.strip() for x in predictNutrients]
def todf(filename):
    data = open("FoodNutrients/"+filename, "r").readlines()
    data = [list(map(float, x.split())) for x in data]
    data = pd.DataFrame(data, columns=nutrients)
    return data

# train = todf("rawFoodDryNutri.txt")
# test = todf("cookedFoodDryNutri.txt")
# y_true = test["Sodium, Na"]

# x_train = train
# y_train = train["Sodium, Na"]

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# test = scaler.transform(test)
# print(test.shape)

# x_train = x_train.reshape(-1, 9, 3, 1)
# test = test.reshape(-1, 9, 3, 1)
# print(x_train)
# print(test.shape)

# x_train, x_val, y_train, y_val=train_test_split(x_train,y_train,test_size=0.15)

# model=Sequential()

# model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(9,3,1)))
# model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2), padding='Same'))
# model.add(Dropout(0.25))

# model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(1,activation='linear'))

# optimizer=RMSprop(learning_rate=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

# model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mse'])

# learning_rate_reduction=ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=1,factor=0.5,min_lr=0.00001)

# epochs = 99
# batch_size = 5

# datagen=ImageDataGenerator(featurewise_center=False, #set input mean to 0 over the data set
#                           samplewise_center=False, #set each sample mean to 0
#                           featurewise_std_normalization=False, #divide inputs by std of the data set
#                           samplewise_std_normalization=False, #divide each input by its std
#                           zca_whitening=False, #apply ZCA whitening
#                           rotation_range=10, #randomly rotate images in the range (degrees, 0 to 180)
#                           zoom_range=0.1, #randomly zoom image
#                           width_shift_range=0.1, #randomly shift images horizontally (fraction of total width)
#                           height_shift_range=0.1, #randomly shifts images vertically (fraction of total height)
#                           horizontal_flip=False, #does not flip images as it could misclassify 6 and 9
#                           vertical_flip=False) #does not flip images as it could misclassify 6 and 9
# datagen.fit(x_train)
# history=model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
#                            epochs=epochs, validation_data=(x_val,y_val),
#                            verbose=2, steps_per_epoch=x_train.shape[0]//batch_size,
#                            callbacks=[learning_rate_reduction])
# model.summary()
# # visualize the structure of CNN
# from keras.utils import plot_model
# # plot diagnostic learning curves
# plt.figure(figsize=(10, 5))

# plot loss during training
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# # plot mse during training
# plt.subplot(212)
# plt.title('Mean Squared Error')
# plt.plot(history.history['mse'], label='train')
# plt.plot(history.history['val_mse'], label='test')
# plt.legend()
# plt.show()
# # evaluate the model
# y_pred = model.predict(test)
# print(y_pred.shape)
# # plot the results
# plt.scatter(y_true, y_pred)
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.show()
# MSE = mean_squared_error(y_true, y_pred)
# MAE = mean_absolute_error(y_true, y_pred)
# R2 = r2_score(y_true, y_pred)
# RMSE = MSE ** 0.5
# SPC = stats.spearmanr(y_true, y_pred)[0]
# print("MSE: ", MSE, "MAE: ", MAE, "R2: ", R2, "RMSE: ", RMSE, "SPC: ", SPC)

def trainCNN(train, test, label):
    y_true = test[label]
    x_train = train
    y_train = train[label]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    test = scaler.transform(test)
    x_train = x_train.reshape(-1, 9, 3, 1)
    test = test.reshape(-1, 9, 3, 1)
    x_train, x_val, y_train, y_val=train_test_split(x_train,y_train,test_size=0.15)
    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(9,3,1)))
    model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), padding='Same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='linear'))
    optimizer=RMSprop(learning_rate=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
    model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mse'])
    learning_rate_reduction=ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
    epochs = 99
    batch_size = 5
    datagen=ImageDataGenerator(featurewise_center=False, #set input mean to 0 over the data set
                          samplewise_center=False, #set each sample mean to 0
                          featurewise_std_normalization=False, #divide inputs by std of the data set
                          samplewise_std_normalization=False, #divide each input by its std
                          zca_whitening=False, #apply ZCA whitening
                          rotation_range=10, #randomly rotate images in the range (degrees, 0 to 180)
                          zoom_range=0.1, #randomly zoom image
                          width_shift_range=0.1, #randomly shift images horizontally (fraction of total width)
                          height_shift_range=0.1, #randomly shifts images vertically (fraction of total height)
                          horizontal_flip=False, #does not flip images as it could misclassify 6 and 9
                          vertical_flip=False) #does not flip images as it could misclassify 6 and 9
    datagen.fit(x_train)
    history=model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
                            epochs=epochs, validation_data=(x_val,y_val),
                            verbose=2, steps_per_epoch=x_train.shape[0]//batch_size,
                            callbacks=[learning_rate_reduction])
    return model, y_true, test, history, x_train

def toSpreadSheet(raw, cooked, name):
    preds = pd.DataFrame()
    true = pd.DataFrame()
    for label in predictNutrients:
        model = trainCNN(todf(raw+ ".txt"), todf(cooked + ".txt"), label)
        y_true = model[1]
        test = model[2]
        y_pred = model[0].predict(test)
        y_pred = pd.DataFrame(y_pred, columns=[label])
        preds = pd.concat([preds, y_pred], axis=1)
        y_true = pd.DataFrame(y_true, columns=[label])
        true = pd.concat([true, y_true], axis=1)
        
        img = model[4]
        CAM(img, model[0], name, label)
        
        model[0].save("CNNmodels/"+ name + label + ".h5")
        history = model[3]
        history = pd.DataFrame(history.history)
        history.to_csv("History/"+ name + label + ".csv", index=False)
    preds.to_csv("CNNpredictions/"+ name + "Predictions.csv", index=False)
    true.to_csv("CNNpredictions/"+ name + "True.csv", index=False)

def CAM(img, model, name, label):
    last_conv_layer = model.layers[1]  
    classifier_layer = model.layers[-1]  
    
    cam_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, classifier_layer.output])

    features, results = cam_model.predict(img)

    cam = np.zeros(dtype=np.float32, shape=features.shape[1:3])
    for i, w in enumerate(results[0]):
        cam += w * features[0, :, :, i]

    cam = cv2.resize(cam, (img.shape[1], img.shape[2]))

    cam /= np.max(cam)
    
    cam = pd.DataFrame(cam)
    cam.to_csv("CNNimportance/"+ name + label + ".csv", index=False)
toSpreadSheet("rawFoodDryNutri", "cookedFoodDryNutri", "dryFood")
# toSpreadSheet("rawFoodWetNutri", "cookedFoodWetNutri", "wetFood")
# feature ranking of my CNN model
# from sklearn.inspection import permutation_importance
# # perform permutation importance
# results = permutation_importance(model, test, y_true, scoring='neg_mean_squared_error')
# # get importance
# importance = results.importances_mean
# # summarize feature importance
# for i,v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()

# split data into train and test sets
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

# # regression me
# regressor = RandomForestRegressor(n_estimators=100)
# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state=0)
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

# def trainModel(X, y, label):
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
#     regressor = DecisionTreeRegressor(random_state=0)
#     regressor.fit(X_train, y_train)
#     y_pred = regressor.predict(X_test)
    
#     MSE = mean_squared_error(y_test, y_pred)
#     MAE = mean_absolute_error(y_test, y_pred)
#     R2 = r2_score(y_test, y_pred)
#     RMSE = MSE ** 0.5
#     SPC = stats.spearmanr(y_test, y_pred)[0]
#     return [label, MSE, MAE, R2, RMSE, SPC]

# def toSpreadSheet(raw, cooked, name):
#     data = []
#     for label in nutrients:
#         data.append(trainModel(todf(raw+ ".txt"), todf(cooked + ".txt")[label], label))
#     data = pd.DataFrame(data, columns=["Nutrient", "MSE", "MAE", "R2", "RMSE", "SPC"])
#     data.to_csv("Evaluations/"+ name + "RandomForest.csv", index=False)
    
# # toSpreadSheet("rawFoodWetNutri")
# toSpreadSheet("rawFoodDryNutri", "cookedFoodDryNutri", "dryFood")
# toSpreadSheet("rawFoodWetNutri", "cookedFoodWetNutri", "wetFood")

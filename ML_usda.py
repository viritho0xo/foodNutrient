
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load your data into X and y arrays
X = np.random.rand(100, 4)  # Example data
y = np.random.rand(100)[:, np.newaxis]  # Example target

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Random Forest regressor with 100 trees
rf = RandomForestRegressor(n_estimators=100)

# Train the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Predict the length output from 4 input features
single_input = np.random.rand(4)  # Example input
single_output = rf.predict([single_input])[0]
print(single_output)

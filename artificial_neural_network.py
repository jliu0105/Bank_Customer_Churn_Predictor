# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
data_scale = StandardScaler()
X_train_val = data_scale.fit_transform(X_train_val)
X_test_val = data_scale.transform(X_test_val)

# Building the ANN

# Initializing the ANN
ANN_model = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ANN_model.add(tf.keras.layers.Dense(units=5, activation='relu'))

# Adding the second hidden layer
ANN_model.add(tf.keras.layers.Dense(units=5, activation='relu'))

# Adding the output layer
ANN_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the ANN

# Compiling the ANN
ANN_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ANN_model.fit(X_train_val, y_train_val, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 550
Gender: Male
Age: 42 years old
Tenure: 3 years
Balance: $ 65000
Number of Products: 3
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 52000

will the customer stay in the bank?

Solution:
"""

print(ANN_model.predict(data_scale.transform([[1, 0, 0, 550, 1, 42, 3, 65000, 3, 1, 1, 52000]])) > 0.5)

# Predicting the Test set results
y_pred = ANN_model.predict(X_test_val)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test_val.reshape(len(y_test_val),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test_val, y_pred)
print(cm)
accuracy_score(y_test_val, y_pred)
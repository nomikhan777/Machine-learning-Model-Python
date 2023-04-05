"""
Importing Libraries
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


"""
1.Reading Stores data from 'storedata.csv'
"""

dataset_df = pd.read_csv("storedata.csv")

# Check columns in dataset
# print(df_file.columns)


"""
2.Cleaning Dataset
"""

# 'Car park' Column contains some invalid values
# Showing Histogram of 'Car park' column before cleaning

plt.hist(dataset_df['Car park'].values)
plt.title('Histogram of "Car park" Column before cleaning')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Cleaning 'Car park' column

dataset_df.replace({'Car park':{'Y':'Yes', 'N': 'No'}}, inplace=True)

# Showing Histogram of 'Car park' column after cleaning

plt.hist(dataset_df['Car park'].values)
plt.title('Histogram of "Car park" Column after cleaning')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()


# Removing outliers and invalid values from 'Staff' column
# print(dataset_df[(dataset_df['Staff'] < 0) | (dataset_df['Staff'] > 9) ])
# 'Staff' columns contains one invalid value and two outliers
dataset_df.drop(dataset_df[dataset_df['Staff'] == -2].index, inplace=True)
dataset_df.drop(dataset_df[dataset_df['Staff'] == 300].index, inplace=True)
dataset_df.drop(dataset_df[dataset_df['Staff'] == 600].index, inplace=True)

"""
3.Preparing training and testing dataset for models
"""

dataset_df.drop(['Town', 'Country', 'Store ID' , 'Manager name'], axis=1, inplace=True)

# Convert categorical variables to numerical using one-hot encoding
dataset_df = pd.get_dummies(dataset_df, columns=['Window', 'Car park', 'Location'])

# Split the dataset into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(dataset_df.drop('Performance', axis=1), 
                                                    dataset_df['Performance'], 
                                                    test_size=0.2, random_state=42)



"""
4. Training Models using Training dataset
"""

# Logistic Regression Model
lr = LogisticRegression()
lr.fit(X_train, Y_train)

# Decision Tree Model
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

# Neural Network Model
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
nn.fit(X_train, Y_train)


"""
5. Evaluation of Models
"""


# Make predictions on the test data
lr_preds = lr.predict(X_test)
dt_preds = dt.predict(X_test)
nn_preds = nn.predict(X_test)


print(lr_preds)
print(dt_preds)
print(nn_preds)

# Calculate the accuracy score for each model
lr_acc = accuracy_score(Y_test, lr_preds)
dt_acc = accuracy_score(Y_test, dt_preds)
nn_acc = accuracy_score(Y_test, nn_preds)

# Print the confusion matrix for each model
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(Y_test, lr_preds))

print("Decision Tree Confusion Matrix:")
print(confusion_matrix(Y_test, dt_preds))

print("Neural Network Confusion Matrix:")
print(confusion_matrix(Y_test, nn_preds))


# Compare the accuracy of each model
print("Logistic Regression Accuracy:", lr_acc)
print("Decision Tree Accuracy:", dt_acc)
print("Neural Network Accuracy:", nn_acc)

# Choose the best performing model
best_model = max(lr_acc, dt_acc, nn_acc)
if best_model == lr_acc:
    print("Logistic Regression is the best performing model.")
elif best_model == dt_acc:
    print("Decision Tree is the best performing model.")
else:
    print("Neural Network is the best performing model.")



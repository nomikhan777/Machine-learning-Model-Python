
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt



dataset_df = pd.read_csv("storedata.csv")



plt.hist(dataset_df['Car park'].values)
plt.title('Histogram of "Car park" Column before cleaning')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()



dataset_df.replace({'Car park':{'Y':'Yes', 'N': 'No'}}, inplace=True)


plt.hist(dataset_df['Car park'].values)
plt.title('Histogram of "Car park" Column after cleaning')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()


dataset_df.drop(dataset_df[dataset_df['Staff'] == -2].index, inplace=True)
dataset_df.drop(dataset_df[dataset_df['Staff'] == 300].index, inplace=True)
dataset_df.drop(dataset_df[dataset_df['Staff'] == 600].index, inplace=True)


dataset_df.drop(['Town', 'Country', 'Store ID' , 'Manager name'], axis=1, inplace=True)


dataset_df = pd.get_dummies(dataset_df, columns=['Window', 'Car park', 'Location'])


X_train, X_test, Y_train, Y_test = train_test_split(dataset_df.drop('Performance', axis=1), 
                                                    dataset_df['Performance'], 
                                                    test_size=0.2, random_state=42)





lr = LogisticRegression()
lr.fit(X_train, Y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)


nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
nn.fit(X_train, Y_train)





lr_preds = lr.predict(X_test)
dt_preds = dt.predict(X_test)
nn_preds = nn.predict(X_test)


print(lr_preds)
print(dt_preds)
print(nn_preds)


lr_acc = accuracy_score(Y_test, lr_preds)
dt_acc = accuracy_score(Y_test, dt_preds)
nn_acc = accuracy_score(Y_test, nn_preds)


print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(Y_test, lr_preds))

print("Decision Tree Confusion Matrix:")
print(confusion_matrix(Y_test, dt_preds))

print("Neural Network Confusion Matrix:")
print(confusion_matrix(Y_test, nn_preds))



print("Logistic Regression Accuracy:", lr_acc)
print("Decision Tree Accuracy:", dt_acc)
print("Neural Network Accuracy:", nn_acc)


best_model = max(lr_acc, dt_acc, nn_acc)
if best_model == lr_acc:
    print("Logistic Regression is the best performing model.")
elif best_model == dt_acc:
    print("Decision Tree is the best performing model.")
else:
    print("Neural Network is the best performing model.")



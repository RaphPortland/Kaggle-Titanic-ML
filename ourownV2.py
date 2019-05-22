


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import pandas as pd
from nnfunctionsV2 import layer_sizes,initialize_parameters, forward_propagation
from nnfunctionsV2 import compute_cost,backward_propagation,update_parameters,nn_model,predict
from sklearn.model_selection import train_test_split

def preprocess(my_data):

    my_data = my_data.drop("Name", axis = 1) # drop the columns named "Name"
    my_data = my_data.drop("PassengerId", axis = 1) # drop the column named "IdPassenger"
    my_data = my_data.drop("Cabin", axis = 1) # drop the column named "Cabin"
    my_data = my_data.drop("Ticket", axis = 1) # drop the column named "Ticket"

    my_data.Age = my_data.Age.fillna(my_data.Age.mean()) # replace NAN values for "Age" by the average Age of the passengers
    my_data.groupby('Embarked').count()  
    my_data.Embarked = my_data.Embarked.fillna('S') # S is the most occures the most in this dataset


    my_data['Embarked'] = pd.Categorical(my_data['Embarked'])
    dfDummies = pd.get_dummies(my_data['Embarked'], prefix = 'category')
    my_data = my_data.join(dfDummies)
    my_data = my_data.drop("Embarked", axis = 1) # drop the column named "Ticket"

    my_data['Sex'] = pd.Categorical(my_data['Sex'])
    dfDummies = pd.get_dummies(my_data['Sex'], prefix = 'Sex_')
    my_data = my_data.join(dfDummies)
    my_data = my_data.drop("Sex", axis = 1) # drop the column named "Ticket"

    my_data.Age = (my_data.Age - my_data.Age.min()) / (my_data.Age.max() - my_data.Age.min())
    my_data.Fare = (my_data.Fare - my_data.Fare.min()) / (my_data.Fare.max() - my_data.Fare.min())

    return my_data;

def reloc(tab, intval):
	tab = tab.values
	tab = tab.T
	if intval>0:
		tab = tab.reshape((1,intval))
	return tab


my_data = pd.read_csv('train.csv', delimiter=',')
TEST = pd.read_csv('test.csv', delimiter=',')
TEST_PASSENGER_ID = TEST.PassengerId

my_data = preprocess(my_data);
TEST = preprocess(TEST)

Y_a = my_data.Survived
X_a = my_data.drop("Survived", axis=1)

x_train, x_test, y_train, y_test = train_test_split(X_a,Y_a)

Y_a = reloc(Y_a, 891)
X_a = reloc(X_a, -1)
x_train = reloc(x_train, -1)
y_train = reloc(y_train, 668)
x_test = reloc(x_test, -1)
y_test = reloc(y_test, 223)
TEST = reloc(TEST, -1)

m = x_train.shape[1]  # training set size
print(m)


print("--------- Starting ----------")

a = {}
#plt.figure(figsize=(16, 32))
#hidden_layer_sizes = [1,2,3,4, 5,6,7,8,9,10,11,12,13,14, 15, 16,17,18,19,20,21,22,23,24,25,40,50]
hidden_layer_sizes = [5,8,9,10,11,12]
#hidden_layer_sizes = [10]
for i, n_h in enumerate(hidden_layer_sizes):

    parameters = nn_model(X_a, Y_a, n_h, num_iterations = 8000, print_cost=True)
    predictions = predict(parameters, x_train)
    predictions_TEST = predict(parameters, x_test)

    accuracy = float((np.dot(y_train,predictions.T) + np.dot(1-y_train,1-predictions.T))/float(y_train.size)*100)
    print ("Accuracy train data for {} hidden units: {} %".format(n_h, accuracy))
    accuracy2 = float((np.dot(y_test,predictions_TEST.T) + np.dot(1-y_test,1-predictions_TEST.T))/float(y_test.size)*100)
    print ("Accuracy test data for {} hidden units: {} %".format(n_h, accuracy2))


    a[str(n_h)]=(accuracy*1 + accuracy2*8)/(9)

    predictions_le_VRAI = predict(parameters, TEST)
    predictions_le_VRAI = pd.DataFrame(predictions_le_VRAI.T)
    predictions_le_VRAI["Survived"] = predictions_le_VRAI[0]
    predictions_le_VRAI = predictions_le_VRAI.drop(0, axis=1)
    predictions_le_VRAI["PassengerId"] = TEST_PASSENGER_ID

    results = predictions_le_VRAI.set_index("PassengerId")
    results["Survived"].replace("False", 0, inplace=True)
    results["Survived"].replace("True", 1, inplace=True)
    results["Survived"] = results["Survived"]*1
    name = "final_" + str(n_h) + ".csv"
    #results.to_csv(name)


b = []
for key,val in a.items():
    print key, "=>", val
    b.append(val)

print(b)




print(a)



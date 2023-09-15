# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\kdata\Desktop\KODI WORK\1. NARESH\2. EVENING BATCH\N_Batch -- 6.00PM\3. JUNE\15th\EMP SAL.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

#Fitting Decision Tree Regression to the datasetmae

from sklearn.tree import DecisionTreeRegressor
"""
Decisions tress (DTs) are the most powerful non-parametric supervised learning method. They can be used for the classification and regression tasks. The main goal of DTs is to create a model predicting target variable value by learning simple decision rules deduced from the data features. Decision trees have two main entities; one is root node, where the data splits, and other is decision nodes or leaves, where we got final output.
What type of decision tree is used in Sklearn?
binary tree algorithm
Sklearn provides a DecisionTreeClassifier and DecisionTreeRegressor classes to build decision tree models. By default, Sklearn's decision tree algorithm uses the CART algorithm, which is a binary tree algorithm that works by recursively partitioning the data into two subsets.
Decision Tree is a decision-making tool that uses a flowchart-like tree structure or is a model of decisions and all of their possible results, including outcomes, input costs, and utility.
Decision-tree algorithm falls under the category of supervised learning algorithms. It works for both continuous as well as categorical output variables.

The branches/edges represent the result of the node and the nodes have either: 

Conditions [Decision Nodes]
Result [End Nodes]
The branches/edges represent the truth/falsity of the statement and take makes a decision based on that in the example below which shows a decision tree that evaluates the smallest of three numbers:  

Decision Tree Regression: 
Decision tree regression observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output. Continuous output means that the output/result is not discrete, i.e., it is not represented just by a discrete, known set of numbers or values.

Discrete output example: A weather prediction model that predicts whether or not there’ll be rain on a particular day. 
Continuous output example: A profit prediction model that states the probable profit that can be generated from the sale of a product.
Here, continuous values are predicted with the help of a decision tree regression model.

Let’s see the Step-by-Step implementation – 


"""
regressor = DecisionTreeRegressor(criterion = 'friedman_mse',splitter = 'random')   
regressor.fit(X, y)

from sklearn.ensemble import RandomForestRegressor 
"""
The sklearn. ensemble module includes two averaging algorithms based on randomized decision trees: the RandomForest algorithm and the Extra-Trees method. Both algorithms are perturb-and-combine techniques [B1998] specifically designed for trees.
What is ensemble Python?
Ensemble means a group of elements viewed as a whole rather than individually. An Ensemble method creates multiple models and combines them to solve it. Ensemble methods help to improve the robustness/generalizability of the model. In this article, we will discuss some methods with their implementation in Python.
What is the use of RandomForestRegressor?
A random forest regressor. A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

"""
reg = RandomForestRegressor(n_estimators = 300, random_state = 0)
reg.fit(X,y)

# Predicting a new result
y_pred = reg.predict([[6.5]])
#now predict previous employee salary & visualize the result
#emplyoee said his salary was 161k but as per dt we got as 150 which was sama as hr call to the X-employee and get that corect information
#what we got in decision tree as 10k less as from previous salary



plt.scatter(X, y, color = 'red')
"""
matplotlib.pyplot.scatter()
Scatter plots are used to observe relationship between variables and uses dots to represent the relationship between them. The scatter() method in the matplotlib library is used to draw a scatter plot. Scatter plots are widely used to represent relation among variables and how change in one affects the other.
"""
plt.plot(X,regressor.predict(X), color = 'blue')
"""
Plotting x and y points
The plot() function is used to draw points (markers) in a diagram.

By default, the plot() function draws a line from point to point.

The function takes parameters for specifying points in the diagram.

Parameter 1 is an array containing the points on the x-axis.

Parameter 2 is an array containing the points on the y-axis.
"""
plt.title('Truth or bluff (Decision tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()
#first part is curve is very good & as i explained this is not a decision tree curve becuase we have to get the tree curve
#algorithm of decission tree is by considering the entrophy and information gain spliting the independent variable into several interval
#as per our tutorial we have 2 independent variable diferent interval forms rectangle & we have to get the averate of independent variable that means alorithm will take interval of algorithm
#you have quastion if you taking average of each interval then how do you have a straight line becuse in decission tree each interval it calculateing the averae of dependent variable
#And you cannot find the average of independent variable & this is not a continuous regression model & the best way to visualize the non-continuous model
#lets plot the higher resolution using tree models


#if you advance visualisation along with tree structure then you will get this resule only
# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#if you check the plot you found the straight & verticle line hear and based on entropy & information gain it splits the whole range in the independent variable to different interval 
"""
What is entropy in Python?
Information entropy quantifies the amount of uncertainty involved in the value of a random variable or the outcome of a random process. Its significance in the decision tree is that it allows us to estimate the impurity or heterogeneity of the target variable.
"""
#if you check the interval of 6 then you get the point of 150k & the range is 5.5. 6o 6.5
#this is all about decission tree regression & for next session we will see the random forest 

















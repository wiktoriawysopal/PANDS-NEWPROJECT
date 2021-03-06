# All the libraries imported which are needed to execute python program
import numpy as np
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['Sepal-length', 'Sepal-width', 'Petal-length', 'Petal-width', 'Class']
dataset = pandas.read_csv(url, names=names)

#Print dimensions of the data set
print(dataset.shape)

#Preview of the first 5 rows of the datase
print(dataset.head(5))

#Check of the last 5 rows of the dataset
print(dataset.tail(5))

#Class distribution
print(dataset.groupby('Class').size())

#Descriptive statistics
print(dataset.describe())

#Histograms generated in Matplot Library
dataset.hist()

plt.show()

#Boxplots generated in Matplot Library
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#Scatter plot matrix
scatter_matrix(dataset)
plt.show()

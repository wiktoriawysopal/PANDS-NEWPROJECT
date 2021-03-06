# PANDS-NEWPROJECT
New repository for programming and scripting project

Wiktoria Wysopal, 23rd of April 2019  
Module: Programming and Scripting - Project  
Degree: Higher Diploma in Data Analytics  
Institute: Galway-Mayo Instituete of Technology, Ireland  

# INSTRUCTIONS:

This repository consist of several files:  
* .gitignore
* LICENSE
* README.md - project documentation
* iris.csv - data set
* iris_code.py - python script
* project.pdf - project description


The data set was downloaded from publicly available source UCI Machine Learning repository: https://archive.ics.uci.edu/ml/datasets/iris. The data was accessed by following tutorial on the machinelearningmastery website. In this case the data was loaded by using website URL. In the first approach the data was downloaded as a .csv file, which is included in this repository however in the further analysis it caused problems.  
Anyone who wants to use this code should get familair with the dataset and project description available in this repository in the pdf file.  
In order to execute the python program the user should download the dataset and install a python program on the computer. The dataset can be downloaded directly from this respository. Also it is recommended to download the Anaconda version with python, since it include needed librarries such as PANDAS, NumPy and Matplot.    
The python script is saved in this folder as iris_code.py. It can be run in any terminal such as built-in powershell in visual studio code or commander. User needs to download the file and save on its own computer. To run the code simply type python iris_code.py in the interpreter and press enter.  


## 1. Background information about the data set

### Who is Ronald Fisher? 
Ronal Aylemer Fisher was one of the greatest scientists of 20th centuary. He specialised in statistics and genetics. He contributed to the world of statistics by creating several methods of analysing the data, such as ANOVA, hypothesis testing and f distribution. He also published several books about statistical research methods, the theory of natural selection and origin of species.   
![Ronald Fisher](https://www.ecured.cu/images/0/09/Sir_Ronald_Fisher_2.jpg)

### What information data set contains?
The data set is from one of the research papers written by Fisher in 1936. In the computer science field it is known as "Iris Flower Data Set". It describes characteristics of iris flower.This data set consist of 150 samples (50 samples from each flower type) The data set contains 4 different attributes. These 4 attributes represent: sepal length, sepal witdh, petal length and petal width. Additionally the data contains of one more column which tells us the type of iris flower, there are 3 different types Setosa, Versicolour and Virginica.)  

![Types of Iris](https://i1.wp.com/dataaspirant.com/wp-content/uploads/2017/01/irises.png?w=600)

### What is the purpose of this data set?
One could think that this data set is for biologist only. Several sources (Ritchieng and Technopedia) say that this data set is famous for machine learning, because it is easy to predict. It is an example of traditional resource which is widely used in computer science for testing purposes.

## 2. Summary of the project investigation
The main aim of this project is to analyse the dataset by using python programming language and writing a python code for it. By doing the research about this data set it has been found that those who analysed this before, focused on calculating basic statistics such as mean, standard deviation, max. and min value of each column. Also the python code for previewving the data was applied as well as class distribution. Therefore the python program for this project calculates the above. 

## 3. Summary of the data set  
The summary of the data set shows the maximum, minimum, mean and standard deviation of each column of the data set.  

|        	| Sepal L  	| Sepal W  	| Petal L  	| Petal W  	|
|--------	|----------	|----------	|----------	|----------	|
| Min.   	| 4.300000 	| 2.000000 	| 1.000000 	| 0.100000 	|
| Max.   	| 7.900000 	| 4.400000 	| 6.900000 	| 2.500000 	|
| Mean   	| 5.84333  	| 3.054000 	| 3.758667 	| 1.198667 	|
| St.dev 	| 0.828066 	| 0.433594 	| 1.764420 	| 0.763161 	|  

## 4. Data Visualization
This section provides graphics generated with the python code (iris_code.py) with the help of the Matplot Library.
### 1.Histograms  
![Figure_1](https://user-images.githubusercontent.com/47478462/56869563-12395b00-69fa-11e9-8fe4-33e172fe7dc3.png)
### 2.Boxplots  
![Figure_2](https://user-images.githubusercontent.com/47478462/56869650-e8ccff00-69fa-11e9-9c94-15bf09ed1a89.png)
### 3.Scatterplot Matrix  
![Figure_3](https://user-images.githubusercontent.com/47478462/56869651-ef5b7680-69fa-11e9-9dc6-09fe63280a61.png)

## 5. Extra information and additional material
While fullfilling this project I came across several difficulties. One of them was problem with commiting my changes to GitHub repository through comander interpreter. I tried the git push command and I had several errors. This project requires reasonable github commit history, even though I tried to solve this error it did not work for me. For this reason I decided to take some screenshots from git log command and combine them into the pdf file, which can be found in this repository, to show the extended commit history.  

Moreover, I found the solution to my project on the last day of the deadline. I downloaded the GitHub Desktop app on my laptop and this helped me to manage all my repositories, commit all the changes and push them into the master repository.  

Finally, the referencing style was not specified. Universities use different references types such as APA or Harvard. For this reason I just included the list of sources I used while aquiring this project, since I am not familiar with referencing method at GMIT.  

## 6. Reference list:
[Famous scientist](https://www.famousscientists.org/ronald-fisher/)  
[Kaggle](https://www.kaggle.com/arshid/iris-flower-dataset)  
[Scikit-learn](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)  
[Ritchieng](https://www.ritchieng.com/machine-learning-iris-dataset/)  
[Techopedia](https://www.techopedia.com/definition/32880/iris-flower-data-set)  
[Machinelearning Mastery](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)  
[Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) 
[Towards Science](https://towardsdatascience.com/introduction-to-data-visualization-in-python-89a54c97fbed)  
[Medium](https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342)  
[Markdow cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)  
[Iris dataset - classification](https://stackoverflow.com/questions/53077801/iris-dataset-machine-learning-classification-model)  
[Table Generator](https://www.tablesgenerator.com/markdown_tables)

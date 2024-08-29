# Using Classifying Algorithm to Determine Safety Level of Cars

This project uses Bohanec,Marko. (1997). Car Evaluation. UCI Machine Learning Repository. https://doi.org/10.24432/C5JP48. This dataset tabulates buying price, maintenance price, number of doors, passenger capacity, trunk size, and estimated safety of cars and gives each car a safety evaluation level, ranging from unacceptable to very good.

The code main.py uses a KNN algorithm with 7 neighbors. However, prior to running the classifying algorithm, categorical variables were encoded using Sklearn's OrdinalEncoder. This simply makes data entries that are text, such as 'high,' 'low,' and 'med' into numbers that can be read by the algorithm. The KNN is able to effectiely classify cars with mean accuracy of 0.92 using the above features.

Random forest classifier and Support Vector Classification (SVC) models were also tested, where random forest yielded much higher accuracy than using the KNN.

The pair most relevant in yielding high accuracy were found by iterating over each possible feature pair and testing the model on the same data. I found that the best feature pair were passenger capacity and estimated safety of cars. While an interesting experiment, using this feature pair only yields an accuracy score of 0.79, which means that conclusive observations cannot be made from using only a pair of features.

Finally, the plotting of decision boundaries was implemented on feature pairs. However these provide very limited insight into the distribution of the data, as the data has only 3-4 discrete values it can take. Nevertheless, the code may be reused in the future to display the decision boundaries models using contiuous data.

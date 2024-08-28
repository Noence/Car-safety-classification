# Using Classifying Algorithm to Determine Safety Level of Cars

This project uses Bohanec,Marko. (1997). Car Evaluation. UCI Machine Learning Repository. https://doi.org/10.24432/C5JP48. This dataset tabulates buying price, maintenance price, number of doors, passenger capacity, trunk size, and estimated safety of cars and gives each car a safety evaluation level, ranging from unacceptable to very good.

The code main.py uses a KNN algorithm with 7 neighbors. However, prior to running the classifying algorithm, categorical variables were encoded using Sklearn's OrdinalEncoder. This simply makes data entries that are text, such as 'high,' 'low,' and 'med' into numbers that can be read by the algorithm. The KNN is able to effectiely classify cars with mean accuracy of 0.92 using the above features.

## Next Steps

Next I would like to plot the decision boundary of the algorithm for different features. Currently the code can do so if it is trained on 2 features rather than 6, yet, the algorithm's accuracy greatly suffers when losing 4 features. Additionally, using other classifying algorithms, such as random forest, or SVMs could be an interesting exercise in improving the model's accuracy. Finally, finding feature importance would give us good insight into how the model makes its decision boundaries.
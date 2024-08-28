from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 
  
# metadata 
print(car_evaluation.metadata) 
  
# variable information 
print(car_evaluation.variables) 

data = pd.read_csv('./data/car_data.csv', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

encoder = OrdinalEncoder()
encoder.fit(data)
print(encoder.categories_)
dataset_encoded = encoder.transform(data)

dataset_encoded = pd.DataFrame(dataset_encoded, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
X = dataset_encoded.drop(columns=['class'])
y = dataset_encoded['class']
# print(encoder.inverse_transform(dataset_encoded))

model = KNeighborsClassifier(n_neighbors=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=34)

model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(acc)

predicted = model.predict(X_test)

# _, ax = plt.subplots(ncols = 1, figsize=(12, 5))


# disp = DecisionBoundaryDisplay.from_estimator(
#     model,
#     X_test,
#     response_method="predict",
#     plot_method="pcolormesh",
#     xlabel='maintenance',
#     ylabel='safety',
#     shading="auto",
#     alpha=0.5,
#     ax=ax,
# )
# scatter = disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k")
# disp.ax_.legend(
#     scatter.legend_elements()[0],
#     encoder.categories_[-1],
#     loc="lower left",
#     title="Classes",
# )

# plt.show()
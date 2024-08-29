from ucimlrepo import fetch_ucirepo 
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


class CarSafetyClassifyer:

    def __init__(self, columns_keep = None):
        # data (as pandas dataframes) 
        self.X = pd.DataFrame 
        self.y = pd.DataFrame
        self.all_data_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        self.data = pd.read_csv('./data/car_data.csv', names=self.all_data_cols)
        self.encode_data(columns_keep)

    def encode_data(self, columns_keep=None):
        if columns_keep is None:
            columns_keep = self.all_data_cols[:-1]
        elif 'class' in columns_keep:
            columns_keep.remove('class')
            print(columns_keep)

        self.encoder = OrdinalEncoder()
        self.encoder.fit(self.data)
        # print(self.encoder.categories_)
        self.dataset_encoded = self.encoder.transform(self.data)

        self.dataset_encoded = pd.DataFrame(self.dataset_encoded, columns=self.all_data_cols)
        self.X = self.dataset_encoded.filter(columns_keep)
        self.y = self.dataset_encoded['class']
        # print(encoder.inverse_transform(dataset_encoded))

    def classify_data(self, model_type='KNN', most_important_pair=False):
        self.model_dict = {
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Random forest': RandomForestClassifier(),
            'SVM': SVC()
        }
        if most_important_pair:
            for pair in combinations(self.all_data_cols[:-1],2):
                self.encode_data(pair)
                print(pair)
                self.classify_data(model_type)
        else:
            self.model = self.model_dict[model_type]

        X_train, self.X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=34)

        self.model.fit(X_train, y_train)
        acc = self.model.score(self.X_test, y_test)
        print(acc)

        predicted = self.model.predict(self.X_test)

    def plot_decision(self, keep_columns=['maint', 'safety']):
        if len(keep_columns)!=2:
            raise IndexError("keep_columns list should have 2 elements")
        
        if len(self.X.columns)!=2:
            self.encode_data(keep_columns)
            self.classify_data()


        _, ax = plt.subplots(ncols = 1, figsize=(12, 5))


        disp = DecisionBoundaryDisplay.from_estimator(
            self.model,
            self.X_test,
            response_method="predict",
            plot_method="pcolormesh",
            xlabel=keep_columns[0],
            ylabel=keep_columns[1],
            shading="auto",
            alpha=0.5,
            ax=ax,
        )
        scatter = disp.ax_.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=self.y, edgecolors="k")
        disp.ax_.legend(
        scatter.legend_elements()[0],
        self.encoder.categories_[-1],
        loc="lower left",
        title="Classes",
        )

        plt.show()




classifier_cars = CarSafetyClassifyer()
classifier_cars.classify_data('KNN', most_important_pair=False)
classifier_cars.plot_decision(['persons', 'safety'])




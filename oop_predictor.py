#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, ensemble
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pikyl
import matplotlib.pyplot as plt

class ESRBClassifier:
    def __init__(self, data_path='games.csv'):
        self.games_data = pd.read_csv(data_path)
        self.feature_columns = None
        self.target_columns = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.best_model = None

    def data_preparation(self):
        self.games_data.drop(['title', 'console'], axis=1, inplace=True)
        self.games_data = pd.get_dummies(self.games_data, columns=['esrb_rating'], prefix='esrb_rating')
        self.games_data = self.games_data.astype(int)
        
        self.feature_columns = self.games_data.drop(['esrb_rating_E', 'esrb_rating_ET', 'esrb_rating_T', 'esrb_rating_M'], axis=1).columns
        self.target_columns = ['esrb_rating_E', 'esrb_rating_ET', 'esrb_rating_T', 'esrb_rating_M']
        
        X = self.games_data[self.feature_columns].to_numpy()
        y = self.games_data[self.target_columns].to_numpy()
        y = np.argmax(y, axis=1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

    def train_model(self, classifier, feature_vector_train, label, feature_vector_valid):
        classifier.fit(feature_vector_train, label)
        predictions = classifier.predict(feature_vector_valid)
        scores = list(metrics.precision_recall_fscore_support(predictions, self.y_test))
        score_vals = [
            scores[0][0],
            scores[1][0],
            scores[2][0],
            metrics.accuracy_score(predictions, self.y_test)
        ]
        return score_vals

    def train_models(self, model, default_params, param_experiments):
        accuracy_compare = {}

        # Default
        accuracy = self.train_model(model(), self.X_train, self.y_train, self.X_test)
        accuracy_compare[model.__name__] = accuracy
        print(model.__name__, accuracy)

        # Experiments
        for i, params in enumerate(param_experiments, start=1):
            accuracy = self.train_model(model(**params), self.X_train, self.y_train, self.X_test)
            accuracy_compare[f"{model.__name__}x{i}"] = accuracy
            print(f"{model.__name__}x{i}", accuracy)

        df_compare = pd.DataFrame(accuracy_compare, index=['precision', 'recall', 'f1 score', 'accuracy'])

        ax = df_compare.plot(kind='bar', legend='reverse')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=len(df_compare.columns))
        plt.ylabel('Score')
        plt.title(f'Comparison of {model.__name__} Models')
        plt.show()

    def train_logistic_regression_models(self):
        default_params = {}
        param_experiments = [
            {'penalty': 'l2', 'C': 0.20, 'random_state': 42},
            {'penalty': 'l2', 'C': 0.25, 'random_state': 42},
            {'penalty': 'l2', 'C': 0.01, 'random_state': 42}
        ]

        self.train_models(linear_model.LogisticRegression, default_params, param_experiments)

    def train_support_vector_machine_models(self):
        default_params = {}
        param_experiments = [
            {'C': 1, 'kernel': 'rbf', 'gamma': 'scale', 'random_state': 42, 'degree': 10},
            {'C': 2.0, 'kernel': 'linear', 'random_state': 42},
            {'C': 1, 'kernel': 'poly', 'degree': 5, 'gamma': 'scale', 'random_state': 42}
        ]

        self.train_models(svm.SVC, default_params, param_experiments)

    def train_random_forest_models(self):
        default_params = {}
        param_experiments = [
            {'n_estimators': 25000, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1,
             'max_features': 'sqrt', 'random_state': 42},
            {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2,
             'max_features': 'log2', 'random_state': 42},
            {'n_estimators': 15000, 'max_depth': 30, 'min_samples_split': 2, 'min_samples_leaf': 2,
             'max_features': 'sqrt', 'random_state': 42}
        ]

        self.train_models(ensemble.RandomForestClassifier, default_params, param_experiments)

    def train_neural_network_model(self):
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'max_iter': [500, 1000, 1500, 2000, 3000]
        }

        mlp = MLPClassifier()
        grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        best_mlp = MLPClassifier(**best_params)
        accuracy = self.train_model(best_mlp, self.X_train, self.y_train, self.X_test)
        print("Neural Network:", accuracy)

        self.best_model = best_mlp

    def export_best_model(self, filename='esrb-model.pkl'):
        with open(filename, 'wb') as file:
            pikyl.dump(self.best_model, file)

if __name__ == "__main__":
    esrb_classifier = ESRBClassifier()
    esrb_classifier.data_preparation()
    esrb_classifier.train_logistic_regression_models()
    esrb_classifier.train_support_vector_machine_models()
    esrb_classifier.train_random_forest_models()
    esrb_classifier.train_neural_network_model()
    esrb_classifier.export_best_model()

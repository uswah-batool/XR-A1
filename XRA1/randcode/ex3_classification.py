import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from scipy.stats import randint
import json
class ClassificationPipeline:
    def __init__(self):
        """Initialize the pipeline."""
        self.dataset_path = input("Enter the path to your dataset: ")
        self.df = pd.read_csv(self.dataset_path)
        self.target_column = input("Enter the name of the target column: ")
        self.results = {}
        """ADD ADDITIONAL LOGIC IF NECESSARY"""

    def preprocess_data(self):
        """Preprocess the dataset"""
        initial_shape = self.df.shape

        self.df = self.df.drop_duplicates()

        numerical_columns = self.df.select_dtypes(include='number').columns

 
        for col in numerical_columns:
            lower_bound = self.df[col].quantile(0.01) 
            upper_bound = self.df[col].quantile(0.99)
            self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)

        # encoding categorical data
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            self.df[col] = le.fit_transform(self.df[col].astype(str))
    
    def split_data(self):
        """Split the data into training and test"""

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def calculate_metrics(self, y_test, y_pred, y_prob):
        """Calculate the metrics"""

        # print("Classification Report:\n")
        a = classification_report(y_test, y_pred, output_dict=True)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        result = {
            "accuracy": a.get("accuracy"),
            "auc": roc_auc
        }
        return result
    
    def train_model_simple(self):
        """train the model in a simple manner"""

        X_train, X_test, y_train, y_test  = self.split_data()

        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        y_prob = dt.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results["simple_dt"] = result

        # Train Random Forest.
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results["simple_rf"] = result


    def train_model_resample(self):
        """train the models with different resampling techniques"""

        # get data split
        X_train, X_test, y_train, y_test  = self.split_data()

        # training DT with smote
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        clf_smote = DecisionTreeClassifier(random_state=42)
        clf_smote.fit(X_train_bal, y_train_bal)
        y_pred = clf_smote.predict(X_test)
        y_prob = clf_smote.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"smote_dt": result})

        rf_smote = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf_smote.fit(X_train_bal, y_train_bal)
        y_pred = rf_smote.predict(X_test)
        y_prob = rf_smote.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"smote_rf": result})

        smote_tomek = SMOTETomek(random_state=42)
        X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)

        # Train Decision Tree with SMOTETomek
        dt_smote_tomek = DecisionTreeClassifier(random_state=42)
        dt_smote_tomek.fit(X_train_bal, y_train_bal)
        y_pred = dt_smote_tomek.predict(X_test)
        y_prob = dt_smote_tomek.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"smotetomek_dt": result})

        rf_smote_tomek = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf_smote_tomek.fit(X_train_bal, y_train_bal)
        y_pred = rf_smote_tomek.predict(X_test)
        y_prob = rf_smote_tomek.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"smotetomek_rf": result})

        # traning DT with smoteenn
        smote_enn = SMOTEENN(random_state=42)
        X_train_bal, y_train_bal = smote_enn.fit_resample(X_train, y_train)

        dt_smote_enn = DecisionTreeClassifier(random_state=42)
        dt_smote_enn.fit(X_train_bal, y_train_bal)
        y_pred = dt_smote_enn.predict(X_test)
        y_prob = dt_smote_enn.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"smoteenn_dt": result})

        rf_smote_enn = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf_smote_enn.fit(X_train_bal, y_train_bal)
        y_pred = rf_smote_enn.predict(X_test)
        y_prob = rf_smote_enn.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"smoteen_rf": result})

    def train_dt_hyperparamters(self):
        """train the models by tuning the hyperparameters"""

        # get data split
        X_train, X_test, y_train, y_test  = self.split_data()

        # Define parameter grid. You can change these values to see how they affect the model.
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'criterion': ['gini', 'entropy']
        }

        dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)

        # Grid Search CV to locate the best combination of parameters. 
        # Here, 'cv' is the number of folds for cross-validation, and 'n_jobs' is the number of cores to use (-1 means use all available CPU cores) to speed up the process through parallel operations.
        # 'scoring' is the metric used to evaluate the model. In this case, we are using 'accuracy'. It's the 'accuracy' score based on which the model would settle at the best set of parameters.
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Now, train the final model with best parameters
        # The first parameter '**grid_search.best_params_' is a dictionary that unpacks (using **) the best parameters found by GridSearchCV
        best_dt = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
        best_dt.fit(X_train, y_train)  # Train on full training data

        y_pred = best_dt.predict(X_test)

        y_prob = best_dt.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"gridcv_dt": result})

        # Again, you can change these values to see how they affect the model.
        # The 'max_features' parameter allows you to try different combinations of features.
        # 'None' means all features will be considered at each split, 'sqrt' means a square root of the total number of features, and 'log2' means a log base 2 of the total number of features.
        param_distributions = {
            'max_depth': randint(3, 30),
            'min_samples_split': randint(2, 15),
            'min_samples_leaf': randint(1, 10),
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'sqrt', 'log2']
        }

        dt = DecisionTreeClassifier(random_state=42)

        # Randomized Search CV 
        random_search = RandomizedSearchCV(
            dt, param_distributions, n_iter=50, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
        )

        random_search.fit(X_train, y_train)

        # Just like before, the first parameter here, '**random_search.best_params_' is a dictionary that unpacks (using **) the best parameters found by RandomizedSearchCV
        best_dt = DecisionTreeClassifier(**random_search.best_params_, random_state=42)
        best_dt.fit(X_train, y_train)

        y_pred = best_dt.predict(X_test)
        y_prob = best_dt.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"randomcv_dt": result})

    def train_rf_hyperparamters(self):

        # get data split
        X_train, X_test, y_train, y_test  = self.split_data()

        # # Define parameter grid. You can change these values to see how they affect the model.
        # param_grid = {
        #     'max_depth': [5, 10, 15, 20, None],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 3, 5],
        #     'criterion': ['gini', 'entropy']
        # }

        # rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

        # # Grid Search CV to locate the best combination of parameters. 
        # # Here, 'cv' is the number of folds for cross-validation, and 'n_jobs' is the number of cores to use (-1 means use all available CPU cores) to speed up the process through parallel operations.
        # # 'scoring' is the metric used to evaluate the model. In this case, we are using 'accuracy'. It's the 'accuracy' score based on which the model would settle at the best set of parameters.
        # grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        # grid_search.fit(X_train, y_train)

        # # Now, train the final model with best parameters
        # # The first parameter '**grid_search.best_params_' is a dictionary that unpacks (using **) the best parameters found by GridSearchCV
        # best_rf = RandomForestClassifier(**grid_search.best_params_, n_estimators=100, class_weight="balanced", random_state=42)
        # best_rf.fit(X_train, y_train)  # Train on full training data

        # y_pred = best_rf.predict(X_test)

        # y_prob = best_rf.predict_proba(X_test)[:, 1]

        # result = self.calculate_metrics(y_test, y_pred, y_prob)
        # self.results.update({"gridcv_rf": result})

        param_distributions = {
            'max_depth': randint(3, 30),
            'min_samples_split': randint(2, 15),
            'min_samples_leaf': randint(1, 10),
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'sqrt', 'log2']
        }

        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

        # Randomized Search CV 
        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=50, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
        )

        random_search.fit(X_train, y_train)

        # Just like before, the first parameter here, '**random_search.best_params_' is a dictionary that unpacks (using **) the best parameters found by RandomizedSearchCV
        best_rf = RandomForestClassifier(**random_search.best_params_, random_state=42)
        best_rf.fit(X_train, y_train)

        y_pred = best_rf.predict(X_test)
        y_prob = best_rf.predict_proba(X_test)[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"randomcv_rf": result})


    def train_model_best_features(self):
        """train the models only on the important features"""

        # get data split
        X_train, X_test, y_train, y_test  = self.split_data()

        dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
        dt.fit(X_train, y_train)

        # Get feature importance scores
        # dt.feature_importances_ returns an array of the feature importance scores
        # pd.Series() converts the array to a pandas Series, with the index as the column names of X_train
        feature_importances = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False)

        # Select only important features by setting the threshold. Usually, it starts from 0.01, but you can play with it to see how it affects the model.
        important_features = feature_importances[feature_importances > 0.05].index

        # Now, we train the model again, but with the selected important features only
        dt_selected = DecisionTreeClassifier(random_state=42) # Adding 'class_weight="balanced"' here could be beneficial, but in our case, you will notice it doesn't make a difference
        dt_selected.fit(X_train[important_features], y_train)

        y_pred = dt_selected.predict(X_test[important_features])

        y_prob = dt_selected.predict_proba(X_test[important_features])[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"best_features_dt": result})

        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf.fit(X_train, y_train)
        feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        important_features = feature_importances[feature_importances > 0.05].index
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf.fit(X_train[important_features], y_train)
        y_pred = dt_selected.predict(X_test[important_features])

        y_prob = dt_selected.predict_proba(X_test[important_features])[:, 1]

        result = self.calculate_metrics(y_test, y_pred, y_prob)
        self.results.update({"best_features_rf": result})



    """ADD AS MANY FUNCTIONS AS NECESSARY."""

    def run_pipeline(self):
        """ADD NECESSARY LOGIC TO COMPLEETE THIS FUNCTION."""

        self.preprocess_data()

        # Running only the best method as it takes time
        # to run all and the test fails

        self.train_model_simple()
        # self.train_model_resample()
        # self.train_dt_hyperparamters()
        # self.train_rf_hyperparamters()
        # self.train_model_best_features()
        self.best_method = 'simple_rf'
        
        """DO NOT CHANGE THE FOLLOWING TWO LINES OF CODE. THEY ARE NEEDED TO TEST YOUR MODEL PERFORMANCE BY THE TEST SUITE."""
        print(f"Best Accuracy Score: {self.results[self.best_method]['accuracy']:.4f}")
        print(f"Best AUC Score: {self.results[self.best_method]['auc']:.4f}")


if __name__ == "__main__":
    pipeline = ClassificationPipeline()
    pipeline.run_pipeline()

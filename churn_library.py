"""
Churn Prediction Library.

This module contains a ChurnPrediction class for data processing,
exploratory data analysis, feature engineering, model training,
and evaluation for churn prediction.
"""

# import libraries
import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, classification_report

from constants import keep_cols, category_lst


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


class ChurnPrediction:
    """Churn prediction class."""

    def __init__(self, path):
        """
        Initialize the class with the data path and load the data.

        input:
            path: a path to the csv file
        """
        self.data_frame = pd.DataFrame()
        self.import_data(path)

    def import_data(self, path):
        """
        Read the csv file found at 'path' and return a data_frame.

        input:
                path: a path to the csv
        output:
                data_frame: pandas data_frame
        """
        self.data_frame = pd.read_csv(path)
        # add churn column
        self.data_frame['Churn'] = self.data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        return self.data_frame

    def perform_eda(self):
        """
        Perform EDA on data_frame and save figures to images folder.

        input:
                None (uses class attribute data_frame)
        output:
                None
        """
        # Churn Distribution
        plt.figure(figsize=(10, 5))
        self.data_frame['Churn'].hist()
        plt.title('Distribution of Churn Among Customers')
        plt.xlabel('Churn (0: No, 1: Yes)')
        plt.ylabel('Frequency of Customers')
        plt.savefig('images/Churn_distribution.png')
        plt.close()

        # Age Distribution
        plt.figure(figsize=(10, 5))
        self.data_frame['Customer_Age'].hist()
        plt.title('Distribution of Customer Age')
        plt.xlabel('Customer Age')
        plt.ylabel('Frequency of Customers')
        plt.savefig('images/Age_distribution.png')
        plt.close()

        # Marital Status Distribution
        plt.figure(figsize=(10, 8))
        self.data_frame['Marital_Status'].value_counts(normalize=True).plot(kind='bar')
        plt.title('Distribution of Marital Status')
        plt.xlabel('Marital Status')
        plt.ylabel('Proportion of Customers')
        plt.savefig('images/Marital_status.png')
        plt.close()

        # Distributions of Total Transaction Count
        plt.figure(figsize=(10, 8))
        sns.histplot(self.data_frame['Total_Trans_Ct'], stat='density', kde=True)
        plt.title('Distribution of Total Transaction Count')
        plt.xlabel('Total Transactions Count')
        plt.ylabel('Density')
        plt.savefig('images/Total_Trans_Ct.png')
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(15, 15))
        sns.heatmap(self.data_frame.corr(), annot=False, cmap='viridis')
        plt.title('Correlation Heatmap of Features')
        plt.savefig('./images/correlation_heatmap.png')
        plt.close()

    def encoder_helper(self, response):
        """
        Encode the categorical columns using churn proportion.

        Group each category by its churn proportion and create
        new column with the churn rate for each category in the
        given data_frame.

        inputs:
            category_lst (list): List of columns that contain
            categorical features.
            response (str): String of the response column name
            (used to calculate churn).

        outputs:
            None (modifies the data_frame)
        """
        temp_df = self.data_frame.copy()

        for category in category_lst:
            category_groups = temp_df.groupby(category).mean()[response]
            temp_df[f'{category}_Churn'] = temp_df[category].map(category_groups)

        self.data_frame = temp_df  # Reassign it back after modification

    def perform_feature_engineering(self):
        """
        Perform feature engineering and data splitting.

        Processe the input data_frame by creating a target variable
        based on customer churn, dropping unnecessary columns, encoding
        categorical features, and scaling numerical features.
        It then splits the data into training and testing sets.

        output:
              x_train, x_test, y_train, y_test
        """
        # Encode the catagorical columns
        self.encoder_helper("Churn")
        # Target column is the churn
        target_data = self.data_frame['Churn']
        # Selected the numerical and decoded columns
        selected_df = pd.DataFrame()
        selected_df[keep_cols] = self.data_frame[keep_cols]

        # Scale the features
        scaler = StandardScaler()
        scaled_feature = scaler.fit_transform(selected_df)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            scaled_feature, target_data, test_size=0.3, random_state=42
        )

        return {
            "x_train_features": x_train,
            "x_test_features": x_test,
            "y_train_targets": y_train,
            "y_test_targets": y_test
        }

    @staticmethod
    def classification_report_image(all_data):
        """
        Generate classification reports and save them as images.

        This function creates classification reports for both the
        training and testing results of logistic regression and
        random forest models. It stores these reports as images
        in the 'images' folder.

        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic
                regression
                y_train_preds_rf: training predictions from random
                forest
                y_test_preds_lr: test predictions from logistic
                regression
                y_test_preds_rf: test predictions from random forest

        output:
                 None
        """
        # access data from dictionary
        y_train = all_data['y_train']
        y_test = all_data['y_test']
        y_train_preds_lr = all_data['y_train_preds_lr']
        y_train_preds_rf = all_data['y_train_preds_rf']
        y_test_preds_lr = all_data['y_test_preds_lr']
        y_test_preds_rf = all_data['y_test_preds_rf']

        # Random Forest Report
        plt.figure(figsize=(8, 10))
        plt.text(0.01, 1.0, 'Random Forest Results', fontsize=14,
                 fontproperties='monospace', transform=plt.gca().transAxes)

        plt.text(0.01, 0.9, 'Train Results:', fontsize=12,
                 fontproperties='monospace', transform=plt.gca().transAxes)
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
                 fontsize=10, fontproperties='monospace', transform=plt.gca().transAxes)

        plt.text(0.01, 0.5, 'Test Results:', fontsize=12, fontproperties='monospace',
                 transform=plt.gca().transAxes)
        plt.text(0.01, 0.3, str(classification_report(y_test, y_test_preds_rf)),
                 fontsize=10, fontproperties='monospace', transform=plt.gca().transAxes)

        plt.axis('off')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust margins
        plt.savefig('./images/rf_classification_report.png')
        plt.close()

        # Logistic Regression Report
        plt.figure(figsize=(8, 10))
        plt.text(0.01, 1.0, 'Logistic Regression Results', fontsize=14,
                 fontproperties='monospace', transform=plt.gca().transAxes)

        plt.text(0.01, 0.9, 'Train Results:', fontsize=12, fontproperties='monospace',
                 transform=plt.gca().transAxes)
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)),
                 fontsize=10, fontproperties='monospace', transform=plt.gca().transAxes)

        plt.text(0.01, 0.5, 'Test Results:', fontsize=12, fontproperties='monospace',
                 transform=plt.gca().transAxes)
        plt.text(0.01, 0.3, str(classification_report(y_test, y_test_preds_lr)),
                 fontsize=10, fontproperties='monospace', transform=plt.gca().transAxes)

        plt.axis('off')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust margins
        plt.savefig('./images/lr_classification_report.png')
        plt.close()

    @staticmethod
    def feature_importance_plot(model, x_data, output_pth):
        """
        Create the feature importances and store in output_pth.

        input:
                model: model object containing feature_importances_
                feature_data: pandas data_frame of feature data
                output_pth: path to store the figure

        output:
                 None
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [keep_cols[i] for i in indices]

        plt.figure(figsize=(20, 15))
        plt.title("Feature Importance")
        plt.bar(range(x_data.shape[1]), importances[indices], align="center")
        plt.xticks(range(x_data.shape[1]), names, rotation=90)
        plt.savefig(output_pth)
        plt.close()

    @staticmethod
    def plot_roc_curves(rfc_model, lr_model, test_data, output_path):
        """
        Plot ROC curves for random forest & logistic regression.

        Parameters:
        - rfc_model: Random Forest model object.
        - lr_model: Logistic Regression model object.
        - X_test: test features.
        - y_test: test target.
        - output_path: path where the ROC curve image will be saved.
        """
        # access x_test and y_test from dictionary
        x_test = test_data['x_test']
        y_test = test_data['y_test']
        # Set up figure for combined ROC curves
        plt.figure(figsize=(10, 8))
        roc_ax = plt.gca()  # Create the axis for plotting

        # Plot ROC for Logistic Regression and Random Forest
        plot_roc_curve(lr_model, x_test, y_test, ax=roc_ax, alpha=0.8)
        plot_roc_curve(rfc_model, x_test, y_test, ax=roc_ax, alpha=0.8)

        # Set plot title
        plt.title('ROC Curves: Random Forest vs Logistic Regression')

        # Save the figure to the provided path
        plt.savefig(output_path)

        # Close the plot to free up memory
        plt.close()

    def train_models(self, x_train, x_test, y_train, y_test):
        """
        Train, store model results (images + scores), and store models.

        input:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        output:
                  None
        """
        # Random Forest
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
        rf_model = RandomForestClassifier(random_state=42)
        cv_rfc = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

        # Logistic Regression
        lr_model = LogisticRegression(solver='lbfgs', max_iter=3000)
        lr_model.fit(x_train, y_train)
        y_train_preds_lr = lr_model.predict(x_train)
        y_test_preds_lr = lr_model.predict(x_test)

        # Save models
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lr_model, './models/lr_model.pkl')

        # Produce classification report images
        all_data = {'y_train': y_train, 'y_test': y_test,
                    'y_train_preds_lr': y_train_preds_lr,
                    'y_train_preds_rf': y_train_preds_rf,
                    'y_test_preds_lr': y_test_preds_lr,
                    'y_test_preds_rf': y_test_preds_rf}
        self.classification_report_image(all_data)

        # Feature importances
        self.feature_importance_plot(cv_rfc.best_estimator_, x_train,
                                     './images/rf_feature_importances.png')
        # ROC curves
        test_data = {'x_test': x_test, 'y_test': y_test}
        self.plot_roc_curves(cv_rfc.best_estimator_, lr_model, test_data,
                             './images/roc_curve.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Churn Prediction Script')
    parser.add_argument('path', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    # Instantiate the ChurnPrediction class with the CSV path
    churn_predict = ChurnPrediction(args.path)

    # Perform EDA
    churn_predict.perform_eda()

    # Perform feature engineering and get train-test split
    data_splits = churn_predict.perform_feature_engineering()
    x_train_features = data_splits["x_train_features"]
    x_test_features = data_splits["x_test_features"]
    y_train_targets = data_splits["y_train_targets"]
    y_test_targets = data_splits["y_test_targets"]

    # Train models
    churn_predict.train_models(x_train_features, x_test_features,
                               y_train_targets, y_test_targets)

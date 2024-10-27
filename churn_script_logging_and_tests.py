"""
Churn Prediction testing and logging script.

This module contains the testing and logging scripts
for the functions in the ChurnPrediction class.
"""

# import libraries
import os
import logging
import argparse
import numpy as np
from churn_library import ChurnPrediction

from constants import category_lst

# Setup logging
logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


class ChurnPredictionTester:
    """Churn prediction tester and logger class."""

    def __init__(self, path):
        """
        Initialize.

        Assigns the path to its own variable,
        and intstantiates the churn prediction class.

        input:
            path: a path to the csv file
        """
        self.path = path  # assign the path as class variable
        self.churn_predictor = ChurnPrediction(path)

    def test_import(self):
        """Test data import."""
        try:
            data_frame = self.churn_predictor.import_data(self.path)
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_data: The file wasn't found")
            raise err

        try:
            assert data_frame.shape[0] > 0
            assert data_frame.shape[1] > 0
            logging.info("Testing import_data: DataFrame shape is %s", data_frame.shape)
        except AssertionError as err:
            logging.error("Testing import_data: File may not have rows & columns")
            raise err

    def test_eda(self):
        """Test perform_eda function."""
        try:
            self.churn_predictor.perform_eda()
            logging.info("Testing perform_eda: SUCCESS")

            # Check if the expected figures are generated
            assert os.path.isfile("./images/Churn_distribution.png")
            assert os.path.isfile("./images/Age_distribution.png")
            assert os.path.isfile("./images/Marital_status.png")
            assert os.path.isfile("./images/Total_Trans_Ct.png")
            assert os.path.isfile("./images/correlation_heatmap.png")
            logging.info("Testing perform_eda: Images saved successfully")
        except AssertionError as err:
            logging.error("Testing perform_eda: EDA images not found")
            raise err
        except Exception as err:
            logging.error("Testing perform_eda: %s", err)
            raise err

    def test_encoder_helper(self):
        """Test encoder_helper function."""
        try:
            self.churn_predictor.encoder_helper("Churn")

            # Check if the new columns are created
            for category in category_lst:
                assert (
                    f"{category}_Churn" in self.churn_predictor.data_frame.columns
                ), f"Encoded column {category}_Churn not created"

            # Verify that the encoded columns contain numeric values
            for category in category_lst:
                assert self.churn_predictor.data_frame[f"{category}_Churn"].dtype in [
                    np.float64,
                    np.float32,
                ], f"{category}_Churn column is not numeric"

            logging.info("Testing encoder_helper: SUCCESS")
        except AssertionError as err:
            logging.error("Testing encoder_helper: %s", err)
            raise err
        except Exception as err:
            logging.error("Testing encoder_helper: %s", err)
            raise err

    def test_perform_feature_engineering(self):
        """Test perform_feature_engineering function."""
        try:
            data_splits = self.churn_predictor.perform_feature_engineering()
            x_train = data_splits["x_train_features"]
            x_test = data_splits["x_test_features"]
            y_train = data_splits["y_train_targets"]
            y_test = data_splits["y_test_targets"]

            # Check if the output datasets have the correct shapes
            assert len(x_train) > 0
            assert len(x_test) > 0
            assert len(y_train) > 0
            assert len(y_test) > 0
            logging.info("Testing perform_feature_engineering: SUCCESS")
        except AssertionError as err:
            logging.error("Testing perform_feature_engineering: Failed to split data")
            raise err
        except Exception as err:
            logging.error("Testing perform_feature_engineering: %s", err)
            raise err

    def test_train_models(self):
        """Test train_models function."""
        try:
            data_splits = self.churn_predictor.perform_feature_engineering()
            x_train = data_splits["x_train_features"]
            x_test = data_splits["x_test_features"]
            y_train = data_splits["y_train_targets"]
            y_test = data_splits["y_test_targets"]

            self.churn_predictor.train_models(x_train, x_test, y_train, y_test)

            # Check if the models are saved and images generated
            assert os.path.isfile("./models/rfc_model.pkl")
            assert os.path.isfile("./models/lr_model.pkl")
            assert os.path.isfile("./images/rf_feature_importances.png")

            logging.info("Testing train_models: SUCCESS")
        except AssertionError as err:
            logging.error("Testing train_models: Models or images not saved properly")
            raise err
        except Exception as err:
            logging.error("Testing train_models: %s", err)
            raise err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Churn Prediction Script')
    parser.add_argument('path', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    # Create an instance of the tester class
    tester = ChurnPredictionTester(args.path)

    # Call the test methods
    tester.test_import()
    tester.test_eda()
    tester.test_encoder_helper()
    tester.test_perform_feature_engineering()
    tester.test_train_models()

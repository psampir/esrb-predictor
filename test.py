import unittest
import os

from oop_predictor import ESRBClassifier

class TestESRBClassifier(unittest.TestCase):
    def setUp(self):
        # Create an instance of ESRBClassifier for testing
        self.esrb_classifier = ESRBClassifier()

    def test_data_preparation(self):
        # Test if data preparation method sets non-empty values for training and testing datasets
        self.esrb_classifier.data_preparation()
        self.assertIsNotNone(self.esrb_classifier.X_train)
        self.assertIsNotNone(self.esrb_classifier.X_test)
        self.assertIsNotNone(self.esrb_classifier.y_train)
        self.assertIsNotNone(self.esrb_classifier.y_test)

    def test_train_logistic_regression_models(self):
        # Test if logistic regression models are trained without errors
        self.esrb_classifier.data_preparation()
        self.esrb_classifier.train_logistic_regression_models()

    def test_train_support_vector_machine_models(self):
        # Test if support vector machine models are trained without errors
        self.esrb_classifier.data_preparation()
        self.esrb_classifier.train_support_vector_machine_models()

    def test_train_random_forest_models(self):
        # Test if random forest models are trained without errors
        self.esrb_classifier.data_preparation()
        self.esrb_classifier.train_random_forest_models()

    def test_train_neural_network_model(self):
        # Test if the neural network model is trained without errors
        self.esrb_classifier.data_preparation()
        self.esrb_classifier.train_neural_network_model()

    def test_export_best_model(self):
        # Test if export of the best-trained model is successful by checking the existence of the exported file
        self.esrb_classifier.data_preparation()
        self.esrb_classifier.train_neural_network_model()
        self.esrb_classifier.export_best_model(filename='test-model.pkl')
        self.assertTrue(os.path.exists('test-model.pkl'))

    def tearDown(self):
        # Clean up any resources if needed
        if os.path.exists('test-model.pkl'):
            os.remove('test-model.pkl')

if __name__ == '__main__':
    # Run the unit tests
    unittest.main()

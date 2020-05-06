import unittest
from unittest.mock import patch, call
from dataset import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset('./resources/dataset/train', './resources/dataset/test', ['0', '1'])

    def test_load_y(self):
        self.dataset._load_y(test=False)
        self.assertEqual(['0', '0', '0', '1', '1'], self.dataset._y_train)

        self.dataset._load_y(test=True)
        self.assertEqual(['0', '0', '1', '1', '1', '1'], self.dataset._y_test)

    @patch('dataset.Dataset._load_x')
    @patch('dataset.Dataset._load_y')
    def test_load_data(self, mock_load_y, mock_load_x):
        self.dataset.load_data()
        mock_load_x.assert_has_calls([call(), call(True)])
        mock_load_y.assert_has_calls([call(), call(True)])


class TestDatasetGrayscaleTrue(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset('./resources/dataset/train', './resources/dataset/test', ['0', '1'], grayscale=True)

    def test_load_x(self):
        self.dataset._load_x(test=False)

        self.assertEqual(5, len(self.dataset.x_train))
        for img in self.dataset.x_train:
            self.assertEqual((28, 28), img.shape)

        self.dataset._load_x(test=True)

        self.assertEqual(6, len(self.dataset.x_test))
        for img in self.dataset.x_train:
            self.assertEqual((28, 28), img.shape)


class TestDatasetGrayscaleFalse(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset('./resources/dataset/train', './resources/dataset/test', ['0', '1'], grayscale=False)

    def test_load_x(self):
        self.dataset._load_x(test=False)

        self.assertEqual(len(self.dataset.x_train), 5)
        for img in self.dataset.x_train:
            self.assertEqual((28, 28, 3), img.shape)

        self.dataset._load_x(test=True)

        self.assertEqual(len(self.dataset.x_test), 6)
        for img in self.dataset.x_train:
            self.assertEqual((28, 28, 3), img.shape)


class TestDatasetGrayscaleResize(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset('./resources/dataset/train',
                               './resources/dataset/test', ['0', '1'],
                               resize_shape=(15, 15))

    def test_load_x(self):
        self.dataset._load_x(test=False)

        self.assertEqual(len(self.dataset.x_train), 5)
        for img in self.dataset.x_train:
            self.assertEqual((15, 15, 3), img.shape)

        self.dataset._load_x(test=True)

        self.assertEqual(len(self.dataset.x_test), 6)
        for img in self.dataset.x_train:
            self.assertEqual((15, 15, 3), img.shape)


class TestDatasetDifferentImageShapes(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset('./resources/different_shapes_dataset/train',
                               './resources/different_shapes_dataset/test',
                               ['0', '1'], grayscale=True)

    def test_load_x(self):
        self.assertRaises(Exception, self.dataset._load_x, True)


if __name__ == '__main__':
    unittest.main()


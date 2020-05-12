import unittest
from unittest.mock import patch, Mock, mock_open, call

import numpy as np

from constructor.API import *


class TestAPIStatic(unittest.TestCase):

    @patch('API.activations.tanh')
    @patch('API.activations.sigmoid')
    @patch('API.activations.relu')
    def test_get_activation(self, mock_relu, mock_sigmoid, mock_tanh):
        res_relu = API.NNConstructorAPI.get_activation('relu')
        res_sigmoid = API.NNConstructorAPI.get_activation('sigmoid')
        res_tanh = API.NNConstructorAPI.get_activation('tanh')

        self.assertEqual(mock_relu, res_relu)
        self.assertEqual(mock_sigmoid, res_sigmoid)
        self.assertEqual(mock_tanh, res_tanh)

    def test_get_activation_exception(self):
        self.assertRaises(ValueError, API.NNConstructorAPI.get_activation, 'something')

    @patch('API.os.makedirs')
    @patch('API.os.path.exists')
    def test_create_model_dir(self, mock_os_path_exists, mock_os_makedirs):
        mock_os_path_exists.return_value = False

        API.NNConstructorAPI._create_model_dir()

        mock_os_path_exists.assert_called_once_with('models')
        mock_os_makedirs.called_once_with('models')

    @patch('API.os.makedirs')
    @patch('API.os.path.exists')
    def test_create_model_dir_not_exists(self, mock_os_path_exists, mock_os_makedirs):
        mock_os_path_exists.return_value = True

        API.NNConstructorAPI._create_model_dir()

        mock_os_path_exists.assert_called_once_with('models')
        mock_os_makedirs.assert_not_called()


class TestAPISetData(unittest.TestCase):
    @patch('API.Model')
    @patch('API.NNConstructorAPI._create_model_dir')
    def setUp(self, _, __):
        self.api = API.NNConstructorAPI()

    @patch('API.Dataset')
    def test_set_data(self, MockDataset):
        mock_dataset_inst = Mock()
        MockDataset.return_value = mock_dataset_inst

        path = './resources/dataset'
        grayscale = False
        resize_shape = False

        self.api.set_data(path, grayscale, resize_shape)

        MockDataset.assert_called_once_with(path + '/train', path + '/test', ['0', '1'], grayscale, resize_shape)
        self.assertEqual(mock_dataset_inst, self.api.dataset)

    def test_set_data_wrong_path(self):
        self.assertRaises(ValueError, self.api.set_data, './resources/wrong_dataset')

    def test_set_data_different_labels(self):
        self.assertRaises(ValueError, self.api.set_data, './resources/different_labels_dataset')


class TestAPISetOptimizer(unittest.TestCase):
    @patch('API.Model')
    @patch('API.NNConstructorAPI._create_model_dir')
    def setUp(self, _, __):
        self.api = API.NNConstructorAPI()

    @patch('API.optimizers')
    def test_set_optimizer(self, mock_optimizers):
        learning_rate = 1
        beta_1 = 2
        beta_2 = 3
        momentum = 4
        rho = 5

        self.api.set_optimizer(algorithm='Adam', learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        mock_optimizers.Adam.assert_called_once_with(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        self.assertIs(mock_optimizers.Adam.return_value, self.api.model.optimizer)

        self.api.set_optimizer(algorithm='SGD', learning_rate=learning_rate, momentum=momentum)
        mock_optimizers.SGD.assert_called_once_with(learning_rate=learning_rate, momentum=momentum)
        self.assertIs(mock_optimizers.SGD.return_value, self.api.model.optimizer)

        self.api.set_optimizer(algorithm='RMSProp', learning_rate=learning_rate, rho=rho)
        mock_optimizers.RMSprop.assert_called_once_with(learning_rate=learning_rate, rho=rho)
        self.assertIs(mock_optimizers.RMSprop.return_value, self.api.model.optimizer)

        self.api.set_optimizer(algorithm='Adagrad', learning_rate=learning_rate)
        mock_optimizers.Adagrad.assert_called_once_with(learning_rate=learning_rate)
        self.assertIs(mock_optimizers.Adagrad.return_value, self.api.model.optimizer)

        self.api.set_optimizer(algorithm='Adadelta', learning_rate=learning_rate, rho=rho)
        mock_optimizers.Adadelta.assert_called_once_with(learning_rate=learning_rate, rho=rho)
        self.assertIs(mock_optimizers.Adadelta.return_value, self.api.model.optimizer)

    def test_set_optimizer_exception(self):
        self.assertRaises(ValueError, self.api.set_optimizer, algorithm='wrong')


class TestAPILayers(unittest.TestCase):
    @patch('API.Model')
    @patch('API.NNConstructorAPI._create_model_dir')
    def setUp(self, _, __):
        self.api = API.NNConstructorAPI()

    @patch('API.Conv2D')
    def test_add_conv(self, MockConv2d):
        filters = 16
        kernel_size = (3, 3)
        activation = 'relu'

        self.api.add_conv(filters, kernel_size, activation)

        MockConv2d.assert_called_once_with(filters=filters, kernel_size=kernel_size)
        self.api.model.add_layer.assert_called_once_with(MockConv2d.return_value)

    @patch('API.Dense')
    def test_add_dense(self, MockDense):
        units = 10
        activation = 'relu'

        self.api.add_dense(units, activation)

        MockDense.assert_called_once_with(units)
        self.api.model.add_layer.assert_called_once_with(MockDense.return_value)

    @patch('API.MaxPooling2D')
    def test_add_max_pooling(self, MockMaxPooling2D):
        pool_size = (3, 3)

        self.api.add_max_pooling(pool_size)

        MockMaxPooling2D.assert_called_once_with(pool_size=pool_size)
        self.api.model.add_layer.assert_called_once_with(MockMaxPooling2D.return_value)

    @patch('API.Flatten')
    def test_add_flatten(self, MockFlatten):
        self.api.add_flatten()

        MockFlatten.assert_called_once()
        self.api.model.add_layer.assert_called_once_with(MockFlatten.return_value)

    @patch('API.Dropout')
    def test_add_dropout(self, MockDropout):
        rate = 1

        self.api.add_dropout(rate)

        MockDropout.assert_called_once_with(rate)
        self.api.model.add_layer.assert_called_once_with(MockDropout.return_value)

    def test_delete_layer(self):
        layer = Mock()

        self.api.delete_layer(layer)

        self.api.model.delete_layer.assert_called_once_with(layer)


class TestAPISaveLoadDelete(unittest.TestCase):
    @patch('API.Model')
    @patch('API.NNConstructorAPI._create_model_dir')
    def setUp(self, _, __):
        self.api = API.NNConstructorAPI()

    @patch('API.os.path.exists')
    @patch('API.os.makedirs')
    def test_save_model(self, mock_os_makedirs, mock_os_path_exists):
        mock_os_path_exists.return_value = False
        name = 'somename'

        self.api.save_model(name)

        mock_os_path_exists.assert_called_once_with('models/' + name)
        mock_os_makedirs.assert_called_once_with('models/' + name)
        self.api.model.save.assert_called_once_with('models/' + name, name)

    @patch('API.os.path.exists')
    @patch('API.os.makedirs')
    def test_save_model_dir_already_exists(self, mock_os_makedirs, mock_os_path_exists):
        mock_os_path_exists.return_value = True
        self.assertRaises(ValueError, self.api.save_model, 'something')
        mock_os_makedirs.assert_not_called()

    @patch('API.os.path.exists')
    @patch('builtins.open', new_callable=mock_open())
    @patch('model.pickle.load')
    def test_load_model(self, mock_pickle_load, mock_o, mock_os_path_exists):
        mock_os_path_exists.return_value = True
        name = 'somename'

        self.api.load_model(name)

        mock_os_path_exists.assert_called_once_with('models/' + name)
        mock_o.assert_called_once_with('models/' + name + '/' + name + '.pkl', 'rb')
        mock_pickle_load.assert_called_once_with(mock_o().__enter__())
        self.api.model.load.assert_called_once_with('models/' + name)

    @patch('API.os.path.exists')
    @patch('builtins.open', new_callable=mock_open())
    @patch('model.pickle.load')
    def test_load_model_exception(self, mock_pickle_load, mock_o, mock_os_path_exists):
        mock_os_path_exists.return_value = False
        name = 'somename'

        self.assertRaises(ValueError, self.api.load_model, name)
        mock_os_path_exists.assert_called_once_with('models/' + name)
        mock_o.assert_not_called()
        mock_pickle_load.assert_not_called()
        self.api.model.load.assert_not_called()

    @patch('API.shutil')
    @patch('API.os.path.exists')
    def test_delete_model(self, mock_os_path_exists, mock_shutil):
        mock_os_path_exists.return_value = True
        name = 'somename'

        self.api.delete_model(name)

        mock_os_path_exists.assert_called_once_with('models/' + name)
        mock_shutil.rmtree.assert_called_once_with('models/' + name)

    @patch('API.shutil')
    @patch('API.os.path.exists')
    def test_delete_model_exception(self, mock_os_path_exists, mock_shutil):
        mock_os_path_exists.return_value = False
        name = 'somename'

        self.assertRaises(ValueError, self.api.delete_model, name)

        mock_os_path_exists.assert_called_once_with('models/' + name)
        mock_shutil.rmtree.assert_not_called()


class TestAPIBuild(unittest.TestCase):
    @patch('API.Model')
    @patch('API.NNConstructorAPI._create_model_dir')
    def setUp(self, _, __):
        self.api = API.NNConstructorAPI()
        self.api.dataset = Mock()
        self.api.dataset.labels = ['1', '2']

    def test_build_zero_layers(self):
        self.api.model.layers = []

        self.assertRaises(ValueError, self.api.build)

    @patch('API.Dense')
    @patch('API.Input')
    def test_build_conv_grayscale(self, MockInput, MockDense):
        layer_1 = Mock(spec=API.Conv2D)
        self.api.model.layers = [layer_1, Mock(), Mock()]
        self.api.dataset.grayscale = True
        x = Mock()
        x.shape = (30, 30)
        self.api.dataset.x_train = [x]

        self.api.build()

        self.api.dataset.load_data.assert_called_once()
        MockInput.assert_called_once_with(shape=(30, 30, 1))
        MockDense.assert_called_once_with(2, activation='softmax')
        self.api.model.add_layer.assert_has_calls([call(MockInput.return_value, 0), call(MockDense.return_value)])
        self.api.model.build.assert_called_once()

    @patch('API.Dense')
    @patch('API.Input')
    def test_build_conv_rgb(self, MockInput, MockDense):
        layer_1 = Mock(spec=API.Conv2D)
        self.api.model.layers = [layer_1, Mock(), Mock()]
        self.api.dataset.grayscale = False
        x = Mock()
        x.shape = (30, 30, 3)
        self.api.dataset.x_train = [x]

        self.api.build()

        self.api.dataset.load_data.assert_called_once()
        MockInput.assert_called_once_with(shape=(30, 30, 3))
        MockDense.assert_called_once_with(2, activation='softmax')
        self.api.model.add_layer.assert_has_calls([call(MockInput.return_value, 0), call(MockDense.return_value)])
        self.api.model.build.assert_called_once()

    @patch('API.Flatten')
    def test_build_dense(self, MockFlatten):
        layer_0 = Mock(spec=API.Dense)
        self.api.model.layers = [layer_0, Mock(), Mock()]

        self.api.build()

        self.api.dataset.load_data.assert_called_once()
        self.api.model.add_layer.assert_has_calls([call(MockFlatten.return_value, 0)])
        self.api.model.build.assert_called_once()


class TestAPIFit(unittest.TestCase):
    @patch('API.Model')
    @patch('API.NNConstructorAPI._create_model_dir')
    def setUp(self, _, __):
        self.api = API.NNConstructorAPI()
        self.api.dataset = Mock()
        self.api.dataset.y_train = ['0', '1']
        self.api.dataset.y_test = ['0', '1']
        self.api.dataset.labels = ['0', '1']
        self.api.dataset.x_train = np.array([[0, 0]])
        self.api.dataset.x_test = np.array([[0, 0]])

    def test_fit_grayscale(self):
        self.api.dataset.grayscale = True

        self.api.fit()

        self.api.model.fit.assert_called_once()

    def test_fit_rgb(self):
        self.api.dataset.grayscale = False

        self.api.fit()

        self.api.model.fit.assert_called_once()


if __name__ == '__main__':
    unittest.main()

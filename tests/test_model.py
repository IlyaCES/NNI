import unittest
from unittest.mock import Mock, patch, mock_open, call
from model import Model, pickle


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.model.name = 'test_name'

    def test_add_layer(self):
        layer_1 = Mock()
        layer_2 = Mock()

        self.model.add_layer(layer_1)
        self.model.add_layer(layer_2)

        self.assertEqual([layer_1, layer_2], self.model.layers)

    @patch('model.load_model')
    def test_load(self, mock_load_model):
        internal_model = Mock()
        layer = Mock()
        internal_model.layers = [layer]
        mock_load_model.return_value = internal_model

        path = ''

        self.model.load(path)

        mock_load_model.assert_called_once_with(path + '/' + self.model.name + '.h5')
        self.assertEqual([layer], self.model.layers)


class TestModelWithLayers(unittest.TestCase):
    def setUp(self):
        layer_1 = Mock()
        layer_2 = Mock()
        self.model = Model()
        self.model.layers = [layer_1, layer_2]

    def test_add_layer_at_index(self):
        layer = Mock()

        self.model.add_layer(layer, 1)

        self.assertEqual(layer, self.model.layers[1])

    def test_delete_layer(self):
        layer = self.model.layers[0]

        self.model.delete_layer(layer)

        self.assertNotIn(layer, self.model.layers)


class TestModelSave(unittest.TestCase):
    def setUp(self):
        self.model = Model()

        self.internal_model = Mock()
        self.model._model = self.internal_model

    @patch('builtins.open', new_callable=mock_open())
    @patch('model.pickle.dump')
    def test_save(self, mock_dump, mock_o):
        path = ''
        name = 'test'

        self.model.save(path, name)

        self.assertIsNone(self.model._model)
        self.assertIsNone(self.model.layers)
        self.internal_model.save.assert_called_once_with(path + '/' + name + '.h5')
        mock_o.assert_called_with(path + '/' + name + '.pkl', 'wb')
        mock_dump.assert_called_once_with(self.model, mock_o().__enter__(), pickle.HIGHEST_PROTOCOL)


class TestModelBuild(unittest.TestCase):
    def setUp(self):
        self.model = Model()

        self.layers = [Mock(), Mock(), Mock()]

        self.mock_add = Mock()
        self.mock_compile = Mock()

        self.internal_model = Mock()
        self.internal_model.add = self.mock_add
        self.internal_model.compile = self.mock_compile

        self.model.layers = self.layers

    @patch('model.optimizers.Adam')
    @patch('model.Sequential')
    def test_build(self, mock_sequential, mock_optimizers_adam):
        mock_sequential.return_value = self.internal_model
        mock_optimizers_adam.return_value = Mock()

        self.model.build()

        mock_sequential.assert_called_once()
        self.mock_add.assert_has_calls(map(call, self.layers), any_order=True)
        self.assertEqual(mock_optimizers_adam.return_value, self.model.optimizer)
        self.model._model.compile.assert_called_once_with(optimizer=mock_optimizers_adam.return_value,
                                                          loss='categorical_crossentropy',
                                                          metrics=['accuracy'])


class TestModelFit(unittest.TestCase):
    def setUp(self):
        self.model = Model()

        self.mock_fit = Mock()

        self.internal_model = Mock()
        self.internal_model.fit = self.mock_fit
        self.internal_model.fit.return_value.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        self.model._model = self.internal_model
        self.model.batch_size = 32
        self.model.epochs = 10

    def test_fit(self):

        train_data = [[1], [2]]
        verbose = 1
        validation_data = Mock()

        self.model.fit(train_data, validation_data)

        self.model._model.fit.assert_called_once_with(*train_data,
                                                      batch_size=self.model.batch_size,
                                                      epochs=self.model.epochs,
                                                      verbose=verbose,
                                                      validation_data=validation_data,
                                                      callbacks=None)


if __name__ == '__main__':
    unittest.main()

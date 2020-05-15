import tkinter as tk
import unittest
from unittest.mock import Mock

from ..test_Gui import TKinterTestCase

class TestModelList(TKinterTestCase):
    def test_list_layers_enter(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Default Enter layer')

    def test_list_layers_exit(self):
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Default Exit layer')

    def test_list_layers(self):
        self.assertIsInstance(self.root.listbox_builder, tk.Listbox)

class TestButtonAdd(TKinterTestCase):
    def test_button_add_layer_on_class(self):
        self.assertIsInstance(self.root.add_layer_button, tk.Button)

    # def test_button_add_layer(self):
    #     self.root.new_layer = Mock()
    #     self.pump_events()
    #     self.root.listbox_builder.select_set(0)
    #     self.pump_events()
    #     self.root.add_layer_button()
    #     self.pump_events()
    #     self.root.new_layer.assert_called_once()

class TestBonusLayer(TKinterTestCase):
    def init_bonus_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()

    def test_bonus_layer_on_class(self):
        self.init_bonus_layer()
        self.assertIsInstance(self.root.layer.listbox_layer, tk.Listbox)

    def test_bonus_layer_convolutional(self):
        self.init_bonus_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(0)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                          'Convolutional')

    def test_bonus_layer_max_pooling(self):
        self.init_bonus_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(1)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                          'MaxPooling')

    def test_bonus_layer_dense(self):
        self.init_bonus_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(2)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                          'Dense')

    def test_bonus_layer_flatten(self):
        self.init_bonus_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(3)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                          'Flatten')

    def test_bonus_layer_dropout(self):
        self.init_bonus_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(4)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                          'Dropout')

class ConvolutionalLayer(TKinterTestCase):
    def init_convolutional_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(0)
        self.pump_events()

    def test_check_filters(self):
        self.init_convolutional_layer()
        self.assertIsInstance(self.root.layer.filters, tk.Entry)

    def test_check_kerne_size_1(self):
        self.init_convolutional_layer()
        self.assertIsInstance(self.root.layer.kernelSize_1, tk.Entry)

    def test_check_kerne_size_2(self):
        self.init_convolutional_layer()
        self.assertIsInstance(self.root.layer.kernelSize_2, tk.Entry)

    def test_check_f_active(self):
        self.init_convolutional_layer()
        self.assertIsInstance(self.root.layer.listbox_conv_layer, tk.Listbox)

class ConvolutionalLayerFilters(TKinterTestCase):
    def init_convolutional_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(0)
        self.pump_events()

    def test_check_filters_init(self):
        self.init_convolutional_layer()
        self.assertEqual(self.root.layer.filters.get(),'32')

    def test_check_filters_error(self):
        self.root.msgError = Mock()
        self.init_convolutional_layer()
        self.root.layer.filters.delete(0, tk.END)
        self.pump_events()
        self.root.layer.filters.insert(0, '0.24')
        self.pump_events()
        self.root.layer.add_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_check_filters_success(self):
        self.init_convolutional_layer()
        self.root.layer.filters.delete(0, tk.END)
        self.pump_events()
        self.root.layer.filters.insert(0, '20')
        self.pump_events()
        self.root.add_Convolutional(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.filters, 20)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

class ConvolutionalLayerKernelSize(TKinterTestCase):
    def init_convolutional_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(0)
        self.pump_events()

    def test_check_kernelSize_1_init(self):
        self.init_convolutional_layer()
        self.assertEqual(self.root.layer.kernelSize_1.get(),'3')

    def test_check_kernelSize_1_error(self):
        self.root.msgError = Mock()
        self.init_convolutional_layer()
        self.root.layer.kernelSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.kernelSize_1.insert(0, '-1')
        self.pump_events()
        self.root.layer.add_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_check_kernelSize_1_success(self):
        self.init_convolutional_layer()
        self.root.layer.kernelSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.kernelSize_1.insert(0, '2')
        self.pump_events()
        self.root.add_Convolutional(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.kernelSize_1, 2)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

    def test_check_kernelSize_2_init(self):
        self.init_convolutional_layer()
        self.assertEqual(self.root.layer.kernelSize_2.get(),'3')

    def test_check_kernelSize_2_error(self):
        self.root.msgError = Mock()
        self.init_convolutional_layer()
        self.root.layer.kernelSize_2.delete(0, tk.END)
        self.pump_events()
        self.root.layer.kernelSize_2.insert(0, '-1')
        self.pump_events()
        self.root.layer.add_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_check_kernelSize_2_success(self):
        self.init_convolutional_layer()
        self.root.layer.kernelSize_2.delete(0, tk.END)
        self.pump_events()
        self.root.layer.kernelSize_2.insert(0, '2')
        self.pump_events()
        self.root.add_Convolutional(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.kernelSize_2, 2)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

class ConvolutionalLayerActive(TKinterTestCase):
    def init_convolutional_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(0)
        self.pump_events()

    def test_check_relu(self):
        self.init_convolutional_layer()
        self.pump_events()
        self.root.layer.listbox_conv_layer.select_set(0)
        self.pump_events()
        self.root.add_Convolutional(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.activations, 'relu')

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

    def test_check_sigmoid(self):
        self.init_convolutional_layer()
        self.pump_events()
        self.root.layer.listbox_conv_layer.select_clear(0, "end")
        self.pump_events()
        self.root.layer.listbox_conv_layer.select_set(1)
        self.pump_events()
        self.root.add_Convolutional(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.activations, 'sigmoid')

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

    def test_check_tanh(self):
        self.init_convolutional_layer()
        self.pump_events()
        self.root.layer.listbox_conv_layer.select_clear(0, "end")
        self.pump_events()
        self.root.layer.listbox_conv_layer.select_set(2)
        self.pump_events()
        self.root.add_Convolutional(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.activations, 'tanh')

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

class MaxPoolingLayer(TKinterTestCase):
    def init_max_pooling_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(1)
        self.pump_events()

    def test_check_pool_size_1(self):
        self.init_max_pooling_layer()
        self.assertIsInstance(self.root.layer.poolSize_1, tk.Entry)

    def test_check_pool_size_2(self):
        self.init_max_pooling_layer()
        self.assertIsInstance(self.root.layer.poolSize_2, tk.Entry)

class MaxPoolingLayerPoolSize(TKinterTestCase):
    def init_max_pooling_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(1)
        self.pump_events()

    def test_check_pool_size_1_init(self):
        self.init_max_pooling_layer()
        self.assertEqual(self.root.layer.poolSize_1.get(),'2')

    def test_check_pool_size_1_error(self):
        self.root.msgError = Mock()
        self.init_max_pooling_layer()
        self.root.layer.poolSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.poolSize_1.insert(0, '-1')
        self.pump_events()
        self.root.layer.add_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_check_pool_size_1_success(self):
        self.init_max_pooling_layer()
        self.root.layer.poolSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.poolSize_1.insert(0, '3')
        self.pump_events()
        self.root.add_MaxPooling(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.poolSize_1, 3)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

    def test_check_pool_size_2_init(self):
        self.init_max_pooling_layer()
        self.assertEqual(self.root.layer.poolSize_2.get(),'2')

    def test_check_pool_size_2_error(self):
        self.root.msgError = Mock()
        self.init_max_pooling_layer()
        self.root.layer.poolSize_2.delete(0, tk.END)
        self.pump_events()
        self.root.layer.poolSize_2.insert(0, '-1')
        self.pump_events()
        self.root.layer.add_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_check_pool_size_2_success(self):
        self.init_max_pooling_layer()
        self.root.layer.poolSize_2.delete(0, tk.END)
        self.pump_events()
        self.root.layer.poolSize_2.insert(0, '3')
        self.pump_events()
        self.root.add_MaxPooling(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.poolSize_2, 3)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

class DenseLayer(TKinterTestCase):
    def init_dense_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(2)
        self.pump_events()

    def test_check_neurons(self):
        self.init_dense_layer()
        self.assertIsInstance(self.root.layer.neurons, tk.Entry)

class DenseLayerNeurons(TKinterTestCase):
    def init_dense_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(2)
        self.pump_events()

    def test_check_neurons_init(self):
        self.init_dense_layer()
        self.assertEqual(self.root.layer.neurons.get(),'64')

    def test_check_neurons_error(self):
        self.root.msgError = Mock()
        self.init_dense_layer()
        self.root.layer.neurons.delete(0, tk.END)
        self.pump_events()
        self.root.layer.neurons.insert(0, '-1')
        self.pump_events()
        self.root.layer.add_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_check_neurons_success(self):
        self.init_dense_layer()
        self.root.layer.neurons.delete(0, tk.END)
        self.pump_events()
        self.root.layer.neurons.insert(0, '3')
        self.pump_events()
        self.root.add_Dense(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.neurons, 3)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

class DropoutLayer(TKinterTestCase):
    def init_dropout_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(3)
        self.pump_events()

    def test_check_neurons(self):
        self.init_dropout_layer()
        self.assertIsInstance(self.root.layer.dropNeurons, tk.Entry)

class DropoutLayerNeurons(TKinterTestCase):
    def init_dropout_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(3)
        self.pump_events()

    def test_check_neurons_init(self):
        self.init_dropout_layer()
        self.assertEqual(self.root.layer.dropNeurons.get(),'0.5')

    def test_check_neurons_error(self):
        self.root.msgError = Mock()
        self.init_dropout_layer()
        self.root.layer.dropNeurons.delete(0, tk.END)
        self.pump_events()
        self.root.layer.dropNeurons.insert(0, '3')
        self.pump_events()
        self.root.layer.add_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_check_neurons_success(self):
        self.init_dropout_layer()
        self.root.layer.dropNeurons.delete(0, tk.END)
        self.pump_events()
        self.root.layer.dropNeurons.insert(0, '0.25')
        self.pump_events()
        self.root.add_Dropout(self.root.layer)
        temp = self.root.layerBuffer[0]
        self.assertEqual(temp.dropNeurons, 0.25)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

class TestButtonDeleteLayer(TKinterTestCase):
    def test_button_delete_layer_on_class(self):
        self.assertIsInstance(self.root.delete_layer_button, tk.Button)

    def test_button_delete_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(3)
        self.pump_events()
        self.root.add_Dropout(self.root.layer)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.assertNotEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Dropout')

class TestButtonChangeLayer(TKinterTestCase):
    def test_button_change_layer_on_class(self):
        self.assertIsInstance(self.root.change_layer_button, tk.Button)

    def test_button_change_layer(self):
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(3)
        self.pump_events()
        self.root.add_Dropout(self.root.layer)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.change_layer_button.invoke()
        self.pump_events()
        self.assertIsInstance(self.root.layer, tk.Toplevel)

        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()

    def test_button_change_layer_enter(self):
        self.root.msgError = Mock()
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.change_layer_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_button_change_layer_exit(self):
        self.root.msgError = Mock()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.change_layer_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

class Test(TKinterTestCase):
    def test_path_browse(self):
        self.assertIsInstance(self.root.path, tk.Entry)

    def test_write(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.path.insert(0, '20'))

class TestOptions(TKinterTestCase):
    def test_options_on_class(self):
        self.assertIsInstance(self.root.listbox_options, tk.Listbox)

    def test_options_adam(self):
        self.pump_events()
        self.root.listbox_options.select_set(0)
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),'Adam')

    def test_options_SGD(self):
        self.pump_events()
        self.root.listbox_options.select_set(1)
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),'SGD')

    def test_options_RMSProp(self):
        self.pump_events()
        self.root.listbox_options.select_set(2)
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),'RMSProp')

    def test_options_adagrad(self):
        self.pump_events()
        self.root.listbox_options.select_set(3)
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),'Adagrad')

    def test_options_adadelta(self):
        self.pump_events()
        self.root.listbox_options.select_set(4)
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),'Adadelta')

class TestOptionsWithArgs(TKinterTestCase):
    def selected_adam(self, learning_rate='0.001', beta_1='0.9', beta_2='0.999'):
        self.pump_events()
        self.root.listbox_options.select_set(0)
        self.pump_events()
        self.root.select_lo_item(self.root)
        self.pump_events()
        self.root.learning_rate.delete(0, tk.END)
        self.pump_events()
        self.root.learning_rate.insert(0, learning_rate)
        self.pump_events()
        self.root.beta_1.delete(0, tk.END)
        self.pump_events()
        self.root.beta_1.insert(0, beta_1)
        self.pump_events()
        self.root.beta_2.delete(0, tk.END)
        self.pump_events()
        self.root.beta_2.insert(0, beta_2)
        self.pump_events()

    def selected_SGD(self, learning_rate='0.01', momentum='0'):
        self.pump_events()
        self.root.listbox_options.select_set(1)
        self.pump_events()
        self.root.select_lo_item(self.root)
        self.pump_events()
        self.root.learning_rate.delete(0, tk.END)
        self.pump_events()
        self.root.learning_rate.insert(0, learning_rate)
        self.pump_events()
        self.root.momentum.delete(0, tk.END)
        self.pump_events()
        self.root.momentum.insert(0, momentum)
        self.pump_events()

    def selected_RMSProp(self, learning_rate='0.001', rho='0.9'):
        self.pump_events()
        self.root.listbox_options.select_set(2)
        self.pump_events()
        self.root.select_lo_item(self.root)
        self.pump_events()
        self.root.learning_rate.delete(0, tk.END)
        self.pump_events()
        self.root.learning_rate.insert(0, learning_rate)
        self.pump_events()
        self.root.rho.delete(0, tk.END)
        self.pump_events()
        self.root.rho.insert(0, rho)
        self.pump_events()

    def selected_adagrad(self, learning_rate='0.01'):
        self.pump_events()
        self.root.listbox_options.select_set(3)
        self.pump_events()
        self.root.select_lo_item(self.root)
        self.pump_events()
        self.root.learning_rate.delete(0, tk.END)
        self.pump_events()
        self.root.learning_rate.insert(0, learning_rate)
        self.pump_events()

    def selected_adadelta(self, learning_rate='1', rho='0.95'):
        self.pump_events()
        self.root.listbox_options.select_set(4)
        self.pump_events()
        self.root.select_lo_item(self.root)
        self.pump_events()
        self.root.learning_rate.delete(0, tk.END)
        self.pump_events()
        self.root.learning_rate.insert(0, learning_rate)
        self.pump_events()
        self.root.rho.delete(0, tk.END)
        self.pump_events()
        self.root.rho.insert(0, rho)
        self.pump_events()

if __name__ == '__main__':
    unittest.main()






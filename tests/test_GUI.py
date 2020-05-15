import tkinter as tk
import unittest
from unittest.mock import Mock
import time

from GUI import NNI


class TKinterTestCase(unittest.TestCase):
    def setUp(self):
        self.root = NNI()
        self.pump_events()

    def tearDown(self):
        if self.root:
            self.root.destroy()
            self.pump_events()

    def pump_events(self):
        while self.root.dooneevent(tk._tkinter.ALL_EVENTS | tk._tkinter.DONT_WAIT):
            pass


class TestGui(TKinterTestCase):

    def test_new_layer_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.new_layer())
        self.pump_events()

    def test_delete_layer_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.delete_layer())
        self.pump_events()

    def test_delete_def_layer_without_selected(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.assertRaises(ValueError, self.root.delete_layer())
        self.pump_events()

    def test_changer_layer_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.change_layer())
        self.pump_events()

    def test_changer_def_layer_without_selected(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.assertRaises(ValueError, self.root.change_layer())
        self.pump_events()

    def test_add_model_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.select_model())
        self.pump_events()

    def test_delete_model_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.delete_model())
        self.pump_events()

    def test_new_layer_convolutional(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(0)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'Convolutional')
        self.pump_events()

    def test_new_layer_maxPooling(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(1)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'MaxPooling')
        self.pump_events()

    def test_new_layer_dense(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(2)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'Dense')
        self.pump_events()

    def test_new_layer_flatten(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(3)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'Flatten')
        self.pump_events()

    def test_new_layer_dropout(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(4)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'Dropout')
        self.pump_events()

    def add_layer_convolutional(self, number_layer, kernel_size_1='3', kernel_size_2='3', filters = "32"):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(0)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
        self.pump_events()
        self.root.layer.kernelSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.kernelSize_1.insert(0, kernel_size_1)
        self.pump_events()
        self.root.layer.kernelSize_2.delete(0, tk.END)
        self.pump_events()
        self.root.layer.kernelSize_2.insert(0, kernel_size_2)
        self.pump_events()
        self.root.layer.filters.delete(0, tk.END)
        self.pump_events()
        self.root.layer.filters.insert(0, filters)
        self.pump_events()
        self.root.add_Convolutional(self.root.layer)
        self.pump_events()

    def add_layer_maxPooling(self, number_layer, pool_size_1='2', pool_size_2='2'):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(1)
        self.pump_events()
        self.root.layer.poolSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.poolSize_1.insert(0, pool_size_1)
        self.pump_events()
        self.root.layer.poolSize_2.delete(0, tk.END)
        self.pump_events()
        self.root.layer.poolSize_2.insert(0, pool_size_2)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
        self.pump_events()
        self.root.add_MaxPooling(self.root.layer)
        self.pump_events()

    def add_layer_dence(self, number_layer, neurons='64'):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(2)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
        self.pump_events()
        self.root.layer.neurons.delete(0, tk.END)
        self.pump_events()
        self.root.layer.neurons.insert(0, neurons)
        self.pump_events()
        self.root.add_Dense(self.root.layer)
        self.pump_events()

    def add_layer_flatten(self, number_layer):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(3)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
        self.pump_events()
        self.root.add_Flatten(self.root.layer)
        self.pump_events()

    def add_layer_dropout(self, number_layer, dropNeurons = "0.5"):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer()
        self.pump_events()
        self.root.layer.listbox_layer.select_set(4)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
        self.pump_events()
        self.root.layer.dropNeurons.delete(0, tk.END)
        self.pump_events()
        self.root.layer.dropNeurons.insert(0, dropNeurons)
        self.pump_events()
        self.root.add_Dropout(self.root.layer)
        self.pump_events()

    def test_add_new_layer_convolutional(self):
        self.pump_events()
        self.add_layer_convolutional(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Convolutional layer')
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_add_new_layer_maxPooling(self):
        self.pump_events()
        self.add_layer_maxPooling(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Max pooling layer')
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_add_new_layer_dence(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Dense layer')
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_add_new_layer_flatten(self):
        self.pump_events()
        self.add_layer_flatten(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Flatten layer')
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_add_new_layer_dropout(self):
        self.pump_events()
        self.pump_events()
        self.add_layer_dropout(0)
        self.pump_events()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.pump_events()
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Dropout layer')
        self.pump_events()
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.pump_events()

    def test_change_layer_convolutional(self):
        self.pump_events()
        self.add_layer_convolutional(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.change_layer()

        self.pump_events()
        self.root.layer.kernelSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.kernelSize_1.insert(0, '10')

        self.pump_events()
        self.root.change_Convolutional(self.root.layer)
        self.pump_events()
        temp = self.root.layerBuffer[0]
        self.pump_events()

        self.assertEqual(temp.kernelSize_1, 10)

        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_change_layer_maxPooling(self):
        self.pump_events()
        self.add_layer_maxPooling(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.change_layer()
        self.pump_events()
        self.root.layer.poolSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.poolSize_1.insert(0, '10')
        self.pump_events()
        self.root.change_MaxPooling(self.root.layer)
        self.pump_events()
        temp = self.root.layerBuffer[0]
        self.pump_events()
        self.assertEqual(temp.poolSize_1, 10)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_change_layer_flaten(self):
        self.pump_events()
        self.add_layer_flatten(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.assertRaises(ValueError, self.root.change_layer())
        self.pump_events()

    def test_change_layer_dence(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.change_layer()
        self.pump_events()
        self.root.layer.neurons.delete(0, tk.END)
        self.pump_events()
        self.root.layer.neurons.insert(0, '6')
        self.pump_events()
        self.root.change_Dense(self.root.layer)
        self.pump_events()
        temp = self.root.layerBuffer[0]
        self.pump_events()
        self.assertEqual(temp.neurons, 6)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_change_layer_dropout(self):
        self.pump_events()
        self.add_layer_dropout(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.change_layer()
        self.pump_events()
        self.root.layer.dropNeurons.delete(0, tk.END)
        self.pump_events()
        self.root.layer.dropNeurons.insert(0, '0.2')
        self.pump_events()
        self.root.change_Dropout(self.root.layer)
        self.pump_events()
        temp = self.root.layerBuffer[0]
        self.pump_events()
        self.assertEqual(temp.dropNeurons, 0.2)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_delete_def_layer(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.delete_layer())
        self.pump_events()

    def test_add_wrong_layer_maxPooling(self):
        self.root.msgError = Mock()
        self.pump_events()
        self.add_layer_maxPooling(0, '0', '-1')
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()

    def test_add_wrong_layer_convolutional_kernal(self):
        self.root.msgError = Mock()
        self.pump_events()
        self.add_layer_convolutional(0,'-2')
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()

    def test_add_wrong_layer_convolutional(self):
        self.root.msgError = Mock()
        self.pump_events()
        self.add_layer_convolutional(0,'3','3','-1')
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()

    def test_add_wrong_layer_dence(self):
        self.root.msgError = Mock()
        self.pump_events()
        self.add_layer_dence(0,'-1')
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()

    def test_add_wrong_layer_dropout(self):
        self.root.msgError = Mock()
        self.pump_events()
        self.add_layer_dropout(0,'-1')
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()

    def test_selected_adam(self, learning_rate='0.001', beta_1='0.9', beta_2='0.999'):
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

    def test_selected_SGD(self, learning_rate='0.01', momentum='0'):
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

    def test_selected_RMSProp(self, learning_rate='0.001', rho='0.9'):
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

    def test_selected_adagrad(self, learning_rate='0.01'):
        self.pump_events()
        self.root.listbox_options.select_set(3)
        self.pump_events()
        self.root.select_lo_item(self.root)
        self.pump_events()
        self.root.learning_rate.delete(0, tk.END)
        self.pump_events()
        self.root.learning_rate.insert(0, learning_rate)
        self.pump_events()

    def test_selected_adadelta(self, learning_rate='1', rho='0.95'):
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

    def test_selected_adam_def(self):
        self.pump_events()
        self.test_selected_adam()
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),
                         'Adam')
        self.pump_events()

    def test_selected_SGD_def(self):
        self.pump_events()
        self.test_selected_SGD()
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),
                         'SGD')
        self.pump_events()

    def test_selected_RMSProp_def(self):
        self.pump_events()
        self.test_selected_RMSProp()
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),
                         'RMSProp')
        self.pump_events()

    def test_selected_adagrad_def(self):
        self.pump_events()
        self.test_selected_adagrad()
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),
                         'Adagrad')
        self.pump_events()

    def test_selected_adadelta_def(self):
        self.pump_events()
        self.test_selected_adadelta()
        self.assertEqual((self.root.listbox_options.get(self.root.listbox_options.curselection())),
                         'Adadelta')
        self.pump_events()

    def test_select_model(self):
        self.pump_events()
        self.root.listbox_folder.select_set(0)
        self.root.select_model()
        self.assertEqual((self.root.listbox_folder.get(self.root.listbox_folder.curselection())),
                         'model')
        self.pump_events()

    def test_start(self):
        self.root.ThreadedTask.run = Mock()
        self.pump_events()
        self.add_layer_flatten(0)
        self.pump_events()
        self.add_layer_maxPooling(0)
        self.pump_events()
        self.add_layer_dropout(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_convolutional(0)
        self.pump_events()
        self.root.path.config(state='normal')
        self.root.path.delete(0, tk.END)
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.root.path.config(state='readonly')
        self.pump_events()
        self.test_selected_SGD()
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.ThreadedTask.run.assert_called_once()
        for i in range(5):
            self.pump_events()
            self.root.listbox_builder.select_set(1)
            self.pump_events()
            self.root.delete_layer()

    def test_start_without_layer(self):
        self.root.msgError = Mock()
        self.root.path.config(state='normal')
        self.root.path.delete(0, tk.END)
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.root.path.config(state='readonly')
        self.pump_events()
        self.test_selected_SGD()
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()

    def test_start_without_path(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.msgError = Mock()
        self.pump_events()
        self.test_selected_SGD()
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_start_wrong_options(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.path.config(state='normal')
        self.root.path.delete(0, tk.END)
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.root.path.config(state='readonly')
        self.pump_events()
        self.root.msgError = Mock()
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_start_wrong_batch(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.path.config(state='normal')
        self.root.path.delete(0, tk.END)
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.root.path.config(state='readonly')
        self.pump_events()
        self.root.msgError = Mock()
        self.pump_events()
        self.root.batch_size.delete(0, tk.END)
        self.pump_events()
        self.root.batch_size.insert(0, "0")
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_start_wrong_rho(self):
        self.root.msgError = Mock()
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.path.config(state='normal')
        self.root.path.delete(0, tk.END)
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.root.path.config(state='readonly')
        self.pump_events()
        self.test_selected_RMSProp('0.001', '-1')
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_start_wrong_epoch(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.path.config(state='normal')
        self.root.path.delete(0, tk.END)
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.root.path.config(state='readonly')
        self.pump_events()
        self.root.msgError = Mock()
        self.pump_events()
        self.root.epochs.delete(0, tk.END)
        self.pump_events()
        self.root.epochs.insert(0, "0")
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_start_wrong_learning_rate(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.path.config(state='normal')
        self.root.path.delete(0, tk.END)
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.root.path.config(state='readonly')
        self.pump_events()
        self.root.msgError = Mock()
        self.pump_events()
        self.test_selected_RMSProp('-1', '1')
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_start_wrong_momentum(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.path.config(state='normal')
        self.root.path.delete(0, tk.END)
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.root.path.config(state='readonly')
        self.pump_events()
        self.root.msgError = Mock()
        self.pump_events()
        self.test_selected_SGD('0.01', '-2')
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_start_wrong_beta_1(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.test_selected_adam('0.001', '-1', '0.999')
        self.pump_events()
        self.root.path.config(state='normal')
        self.pump_events()
        self.root.path.delete(0, tk.END)
        self.pump_events()
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.pump_events()
        self.root.path.config(state='readonly')
        self.pump_events()
        self.root.msgError = Mock()
        self.pump_events()
        self.root.start()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer()
        self.pump_events()

    def test_start_wrong_beta_2(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.test_selected_adam('0.001', '0.9', '-1')
        self.pump_events()
        self.root.path.config(state='normal')
        self.root.path.delete(0, tk.END)
        self.root.path.insert(0, 'C:/Users/ready/dev/mnist_test')
        self.root.path.config(state='readonly')
        self.pump_events()
        self.root.msgError = Mock()
        self.pump_events()
        self.root.start_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer_button.invoke()
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.delete_layer_button.invoke()
        self.pump_events()

    def test_save_model_without_name(self):
        self.root.save_button.config(state='normal')
        self.root.msgError = Mock()
        self.pump_events()
        self.root.save_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_save_model_with_name_mo25(self):
        self.root.save_button.config(state='normal')
        self.root.msgError = Mock()
        self.pump_events()
        self.root.name.insert(0, "11111111111111111111111111111")
        self.pump_events()
        self.root.save_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()

    def test_save_model(self):
        self.root.save_button.config(state='normal')
        self.root.constructorAPI.save_model = Mock()
        self.pump_events()
        self.root.name.insert(0, "Model_11")
        self.pump_events()
        self.root.save_button.invoke()
        self.pump_events()
        self.root.constructorAPI.save_model.assert_called_once()

    def test_delete_model(self):
        self.root.constructorAPI.delete_model = Mock()
        self.pump_events()
        self.root.listbox_folder.select_set(0)
        self.pump_events()
        self.root.delete_model_button.invoke()
        self.pump_events()
        self.root.constructorAPI.delete_model.assert_called_once()

    def test_browse(self):
        tk.filedialog.askdirectory = Mock()
        self.root.browse_button.invoke()
        self.pump_events()
        tk.filedialog.askdirectory.assert_called_once()

    def test_new_no_layer(self):
        self.root.msgError = Mock()
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.add_layer_button.invoke()
        self.pump_events()
        self.root.layer.add_button.invoke()
        self.pump_events()
        self.root.msgError.assert_called_once()
        self.pump_events()

if __name__ == '__main__':
    unittest.main()

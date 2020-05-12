import tkinter as tk
import unittest

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

class test_gui(TKinterTestCase):

    #####Testing_options

    def test_batch_saze(self):
        self.pump_events()
        self.root.batch_size.delete(0, tk.END)
        self.pump_events()
        self.root.batch_size.insert(0, "0")
        self.pump_events()
        self.assertEqual(self.root.batch_size.get(), "0")

    #####Test_no_selected_item

    def test_new_layer_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.new_layer(self.root))
        self.pump_events()
    def test_delete_layer_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.delete_layer(self.root))
        self.pump_events()
    def test_delete_def_layer_without_selected(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.assertRaises(ValueError, self.root.delete_layer(self.root))
        self.pump_events()
    def test_changer_layer_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.change_layer(self.root))
        self.pump_events()
    def test_changer_def_layer_without_selected(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.assertRaises(ValueError, self.root.change_layer(self.root))
        self.pump_events()
    def test_add_model_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.select_model(self.root))
        self.pump_events()
    def test_delete_model_without_selected(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.delete_model(self.root))
        self.pump_events()

    #####test_add_layer

    def test_new_layer_convolutional(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(0)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'Convolutional')
        self.pump_events()
    def test_new_layer_maxPooling(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(1)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'MaxPooling')
        self.pump_events()
    def test_new_layer_dense(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(2)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'Dense')
        self.pump_events()
    def test_new_layer_flatten(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(3)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'Flatten')
        self.pump_events()
    def test_new_layer_dropout(self):
        self.pump_events()
        self.root.listbox_builder.select_set(0)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(4)
        self.assertEqual((self.root.layer.listbox_layer.get(self.root.layer.listbox_layer.curselection())),
                         'Dropout')
        self.pump_events()

    #####Add_layer

    def add_layer_convolutional(self, number_layer):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(0)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
        self.pump_events()
        self.root.add_Convolutional(self.root.layer)
        self.pump_events()
    def add_layer_maxPooling(self, number_layer, poolSize_1='2', poolSize_2='2'):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(1)
        self.pump_events()
        self.root.layer.poolSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.poolSize_1.insert(0, poolSize_1)
        self.pump_events()
        self.root.layer.poolSize_2.delete(0, tk.END)
        self.pump_events()
        self.root.layer.poolSize_2.insert(0, poolSize_2)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
        self.pump_events()
        self.root.add_MaxPooling(self.root.layer)
        self.pump_events()
    def add_layer_dence(self, number_layer):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(2)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
        self.pump_events()
        self.root.add_Dense(self.root.layer)
        self.pump_events()
    def add_layer_flatten(self, number_layer):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(3)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
        self.pump_events()
        self.root.add_Flatten(self.root.layer)
        self.pump_events()
    def add_layer_dropout(self, number_layer):
        self.pump_events()
        self.root.listbox_builder.select_set(number_layer)
        self.pump_events()
        self.root.new_layer(self.root)
        self.pump_events()
        self.root.layer.listbox_layer.select_set(4)
        self.pump_events()
        self.root.select_layer(self.root.layer, self.root)
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
        self.root.delete_layer(self.root)
        self.pump_events()
    def test_add_new_layer_maxPooling(self):
        self.pump_events()
        self.add_layer_maxPooling(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Max pooling layer')
        self.pump_events()
        self.root.delete_layer(self.root)
        self.pump_events()
    def test_add_new_layer_dence(self):
        self.pump_events()
        self.add_layer_dence(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Dense layer')
        self.pump_events()
        self.root.delete_layer(self.root)
        self.pump_events()
    def test_add_new_layer_flatten(self):
        self.pump_events()
        self.add_layer_flatten(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.assertEqual((self.root.listbox_builder.get(self.root.listbox_builder.curselection())),
                         'Flatten layer')
        self.pump_events()
        self.root.delete_layer(self.root)
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
        self.root.delete_layer(self.root)
        self.pump_events()
    def test_delete_def_layer(self):
        self.pump_events()
        self.assertRaises(ValueError, self.root.delete_layer(self.root))
        self.pump_events()
    def test_add_wrong_layer_maxPooling(self):
        self.pump_events()
        self.assertRaises(ValueError, self.add_layer_maxPooling(0, '0', '0'))
        self.pump_events()

#####def_for_select_io_item

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

    #####Test_select_item

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
        self.root.select_model(self.root)
        self.assertEqual((self.root.listbox_folder.get(self.root.listbox_folder.curselection())),
                         'model')
        self.pump_events()

    def test_change_layer_convolutional(self):
        self.pump_events()
        self.add_layer_convolutional(0)
        self.pump_events()
        self.root.listbox_builder.select_set(1)
        self.pump_events()
        self.root.change_layer(self.root)

        self.pump_events()
        self.root.layer.kernelSize_1.delete(0, tk.END)
        self.pump_events()
        self.root.layer.kernelSize_1.insert(0, '10')

        self.pump_events()
        self.root.change_Convolutional(self.root.layer)
        self.pump_events()
        temp = self.root.layerBuffer[0]
        self.pump_events()

        self.assertEqual(temp.kernelSize_1,10)

        self.pump_events()
        self.root.delete_layer(self.root)
        self.pump_events()

    # def test_change_layer_(self):
    #     self.pump_events()
    #     self.add_layer_maxPooling(0)
    #     self.pump_events()
    #     self.root.listbox_builder.select_set(1)
    #     self.pump_events()
    #     self.root.change_layer(self.root)
    #
    #     self.pump_events()
    #     self.root.layer.kernelSize_1.delete(0, tk.END)
    #     self.pump_events()
    #     self.root.layer.kernelSize_1.insert(0, '10')
    #
    #     self.pump_events()
    #     self.root.change_Convolutional(self.root.layer)
    #     self.pump_events()
    #     temp = self.root.layerBuffer[0]
    #     self.pump_events()
    #
    #     self.assertEqual(temp.kernelSize_1,10)
    #
    #     self.pump_events()
    #     self.root.delete_layer(self.root)
    #     self.pump_events()



if __name__ == '__main__':
    unittest.main()
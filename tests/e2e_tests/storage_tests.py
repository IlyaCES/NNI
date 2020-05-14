import os
import sys
import unittest
from pathlib import Path
from shutil import rmtree
from unittest.mock import Mock

from ..test_GUI import TKinterTestCase


class TestModelLoad(TKinterTestCase):
    def test_load_model(self):
        self.pump_events()
        self.root.listbox_folder.select_set(0)
        self.pump_events()

        self.root.select_model('<Button-1>')

        self.pump_events()

        self.assertTrue(hasattr(self.root, 'accuracy_plot'))
        self.assertTrue(hasattr(self.root, 'loss_plot'))
        self.assertNotEquals(0, self.root.log.get(1.0, 'end'))

    def test_load_model_already_has_plots(self):
        # Load model
        self.root.listbox_folder.select_set(0)
        self.pump_events()
        self.root.select_model('<Button-1>')

        text = self.root.log.get(1.0, 'end')
        loss_plot = self.root.loss_plot
        accuracy_plot = self.root.accuracy_plot

        self.root.listbox_folder.select_clear(0)

        # Load different model second time. Plots and text should be reloaded
        self.pump_events()
        self.root.listbox_folder.select_set(1)
        self.pump_events()

        self.root.select_model('<Button-1>')

        self.pump_events()

        self.assertNotEqual(self.root.loss_plot, loss_plot)
        self.assertNotEqual(self.root.accuracy_plot, accuracy_plot)
        self.assertNotEqual(self.root.log.get(1.0, 'end'), '\n')
        self.assertNotEqual(self.root.log.get(1.0, 'end'), text)


class TestDeleteModel(TKinterTestCase):
    def setUp(self):
        os.mkdir('models/test_model')
        Path('models/test_model/test_model.pkl').touch()
        Path('models/test_model/test_model.h5').touch()
        TKinterTestCase.setUp(self)

    def tearDown(self):
        if Path('models/test_model').exists():
            rmtree('models/test_model')
        TKinterTestCase.tearDown(self)

    def test_delete_model(self):
        self.pump_events()
        model_names = self.root.listbox_folder.get(0, 'end')
        for i, name in enumerate(model_names):
            if name == 'test_model':
                self.root.listbox_folder.select_set(i)
                break
        self.pump_events()
        self.root.delete_model('<Button-2>')

        self.assertFalse(Path('models/test_model/test_model.pkl').exists())
        self.assertFalse(Path('models/test_model/test_model.h5').exists())

    def test_delete_model_not_selected(self):
        self.pump_events()
        self.root.msgError = Mock()

        self.root.delete_model('<Button-2>')

        self.root.msgError.assert_called_once()


class TestModelList(TKinterTestCase):
    def test_select_model_list(self):
        models_list = os.listdir('models')
        self.assertEqual(list(self.root.listbox_folder.get(0, 'end')), models_list)


if __name__ == '__main__':
    unittest.main()

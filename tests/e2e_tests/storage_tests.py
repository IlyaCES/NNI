import unittest
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


if __name__ == '__main__':
    unittest.main()

import pickle
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


model = None  # model брать из API
with open('./models/test_save_2/test_save_2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
#plt.ion()

fig, (accuracy_plot, loss_plot) = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
x = range(1, len(model.accuracy) + 1)
ap_1, = accuracy_plot.plot([], [], label='Training')
accuracy_plot.plot(x, model.val_accuracy, label='Validation')
accuracy_plot.legend()
accuracy_plot.set_xlabel('Epochs')
accuracy_plot.set_ylabel('Accuracy')
accuracy_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
accuracy_plot.set_xlim(1, len(x))

loss_plot.plot(x, model.loss, label='Training')
loss_plot.plot(x, model.val_loss, label='Validation')
loss_plot.legend()
loss_plot.set_xlabel('Epochs')
loss_plot.set_ylabel('Loss')
loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
loss_plot.set_xlim(1, len(x))
plt.close()

for i in range(100):
    print(i, len(range(1, i+2)), len(model.accuracy[:i + 1]))
    ap_1.set_data(range(1, i+2), model.accuracy[:i+1])
    plt.pause(0.02)
    plt.close()
    #fig.canvas.draw()
    #fig.canvas.flush_events()
#plt.ioff()
plt.show()
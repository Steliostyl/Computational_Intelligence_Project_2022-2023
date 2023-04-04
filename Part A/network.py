import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import gradient_descent_v2
from keras.metrics import Accuracy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std

H = 0.01  # Learning rate.
M = 0.6  # Momentum
NW_IN = 22  # Size of the input of the network.
NW_OUT = 5  # Output of the network (labels). Equal to the ammount of classes.
EPOCHS = 10  # Epochs in training
BATCH_SIZE = 20  # Batch size
metrics = ["MSE", "binary_accuracy"]
h_metrics = metrics + ["loss"]
val_metrics = ["val_" + metric for metric in h_metrics]

def getNetworkInput(processed_df: pd.DataFrame) -> list:
  """Accepts the processed dataset dataframe as input
  and returns a tuple of X, y used to train the model later."""

  class_categories = [
      'class_sitting', 'class_sittingdown', 'class_standing',
      'class_standingup', 'class_walking'
  ]

  X = processed_df.drop(class_categories, axis=1)
  y = processed_df[class_categories]

  return X.values, y.values

def getModel():
  model = Sequential()
  model.add(
      Dense(NW_IN + NW_OUT, input_dim=NW_IN, kernel_initializer='he_uniform',
            activation='relu'))
  model.add(Dense(NW_OUT, activation='sigmoid'))
  model.compile(loss='binary_crossentropy',
                optimizer=gradient_descent_v2.SGD(learning_rate=H,
                                                  momentum=M), metrics=metrics)
  #model.compile(
  #    loss='binary_crossentropy', optimizer='adam',
  #    metrics=['binary_accuracy', 'MSE'])
  return model

def evaluateModel(X, y) -> tuple[object, pd.DataFrame]:
  cv = KFold(n_splits=5, shuffle=True)
  final_metrics_list = []
  best_acc = 0
  for i, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = getModel()
    history = model.fit(X_train, y_train, verbose=1, epochs=EPOCHS,
                        validation_data=(X_test, y_test))
    final_metrics_list.append(
        [i + 1] +
        [history.history[metric][-1] for metric in h_metrics + val_metrics])

    val_bin_acc = history.history['val_binary_accuracy'][-1]
    if val_bin_acc > best_acc:
      best_history = history
      best_acc = val_bin_acc

  final_metrics = pd.DataFrame(final_metrics_list, columns=["Fold"] +
                               h_metrics + val_metrics).set_index('Fold')
  final_metrics.loc["Average"] = final_metrics.mean()
  best_row_index = final_metrics['val_loss'].idxmin()
  final_metrics.loc["Best"] = final_metrics.loc[best_row_index]

  return (best_history, final_metrics)

def plot_result(history, item):
  """Used for the graphs. Takes as input the name of the metric and 
    outputs graphs with the progression of said metric over EPOCHS."""

  plt.plot(history.history[item], label=item)
  plt.plot(history.history["val_" + item], label="val_" + item)
  plt.xlabel("Epochs")
  plt.ylabel(item)
  plt.title("Train and validation {} over epochs".format(item), fontsize=14)
  plt.legend()
  plt.grid()
  plt.show()

def plot_history(history) -> None:
  items = h_metrics

  [plot_result(history, item) for item in items]
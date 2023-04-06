from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import gradient_descent_v2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from data_prep import CLASSES

H = 0.001  # Learning rate.
M = 0.0  # Momentum
NW_IN = 22  # Size of the input of the network.
NW_OUT = 5  # Output of the network (labels). Equal to the ammount of classes.
EPOCHS = 20  # Epochs in training
BATCH_SIZE = 10  # Batch size
metrics = ["MSE", "accuracy"]
h_metrics = metrics + ["loss"]
val_metrics = ["val_" + metric for metric in h_metrics]

def getNetworkInput(processed_df: pd.DataFrame) -> list:
  """Accepts the processed dataset dataframe as input
  and returns a tuple of X, y used to train the model later."""

  X = processed_df.drop(CLASSES, axis=1)
  y = processed_df[CLASSES]

  return X.values, y.values

def getModel():
  model = Sequential()
  model.add(Dense(NW_IN + NW_OUT, input_dim=NW_IN, activation='relu'))
  model.add(Dense(NW_OUT, activation='softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer=gradient_descent_v2.SGD(learning_rate=H,
                                                  momentum=M), metrics=metrics)
  return model

def evaluateModel(X, y) -> list[dict, pd.DataFrame]:
  cv = KFold(n_splits=5, shuffle=True)
  final_metrics_list = []
  best_loss = 999999
  for i, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = getModel()
    history = model.fit(X_train, y_train, verbose=1, epochs=EPOCHS,
                        validation_data=(X_test, y_test))
    final_metrics_list.append(
        [i + 1] +
        [history.history[metric][-1] for metric in h_metrics + val_metrics])

    val_loss = history.history['val_loss'][-1]
    if val_loss < best_loss:
      best_history = history.history
      best_loss = val_loss

  final_metrics = pd.DataFrame(final_metrics_list, columns=["Fold"] +
                               h_metrics + val_metrics).set_index('Fold')
  final_metrics.loc["Average"] = final_metrics.mean()
  best_row_index = final_metrics['val_loss'].idxmin()
  final_metrics.loc["Best"] = final_metrics.loc[best_row_index]

  return [best_history, final_metrics]

def plot_result(history, item):
  """Used for the graphs. Takes as input the name of the metric and 
    outputs graphs with the progression of said metric over EPOCHS."""

  plt.plot(history[item], label=item)
  plt.plot(history["val_" + item], label="val_" + item)
  plt.xlabel("Epochs")
  plt.ylabel(item)
  plt.title("Train and validation {} over epochs".format(item), fontsize=14)
  plt.legend()
  plt.grid()
  plt.show()

def plot_history(history) -> None:
  items = h_metrics

  [plot_result(history, item) for item in items]
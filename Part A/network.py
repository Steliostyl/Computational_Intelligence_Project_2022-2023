from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import gradient_descent_v2, adam_v2
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from data_prep import CLASSES

H = 0.05  # Learning rate.
M = 0.6  # Momentum
NW_IN = 22  # Size of the input of the network.
NW_OUT = 5  # Output of the network (labels). Equal to the ammount of classes.
EPOCHS = 100  # Epochs in training
R = 0.0  # Regulization factor
metrics = ["MSE", "accuracy"]
h_metrics = metrics + ["loss"]
val_metrics = ["val_" + metric for metric in h_metrics]

def getNetworkInput(processed_df: pd.DataFrame) -> tuple:
  """Accepts the processed dataset dataframe as input
  and returns a tuple of X, y used to train the model later."""

  X = processed_df.drop(CLASSES, axis=1)
  y = processed_df[CLASSES]

  print(X.shape, y.shape)

  return X.values, y.values

def getModel() -> Sequential:
  """Defines the model and returns it."""

  model = Sequential()
  model.add(
      Dense(NW_IN + NW_OUT, input_dim=NW_IN, activation='relu',
            kernel_regularizer=l2(R), bias_regularizer=l2(R)))
  model.add(
      Dense((NW_IN + NW_OUT) // 2, activation='relu', kernel_regularizer=l2(R),
            bias_regularizer=l2(R)))
  model.add(
      Dense((NW_IN + NW_OUT) // 3, activation='relu', kernel_regularizer=l2(R),
            bias_regularizer=l2(R)))
  model.add(Dense(NW_OUT, activation='softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer=gradient_descent_v2.SGD(learning_rate=H,
                                                  momentum=M), metrics=metrics)
  return model

def evaluateModel(X, y) -> list[dict, pd.DataFrame]:
  """Trains and evaluates a model on stance prediction of an individual
  based on sensor data."""

  early_stopping = EarlyStopping(monitor="val_accuracy", patience=10,
                                 mode="max", min_delta=0.001,
                                 restore_best_weights=True)

  # Define k-fold
  cv = KFold(n_splits=5, shuffle=True)
  final_metrics_list = []
  best_loss = 999999

  # Run the 5 folds
  for i, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = getModel()
    # Fit the model to the train data and evaluate it with the test data
    history = model.fit(X_train, y_train, verbose=1, epochs=EPOCHS,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    # Append this fold's metrics to the final_metrics_list
    final_metrics_list.append([i + 1] + [
        max(history.history[metric]) if metric.
        endswith('accuracy') else min(history.history[metric])
        for metric in h_metrics + val_metrics
    ])

    # Save best model's history to plot later
    val_loss = history.history['val_loss'][-1]
    if val_loss < best_loss:
      best_history = history.history
      best_loss = val_loss

  # Create a table containing the metrics of each fold,
  # along with the average and best model's metrics.
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
  """Plots all the metrics in given history."""

  items = h_metrics

  [plot_result(history, item) for item in items]

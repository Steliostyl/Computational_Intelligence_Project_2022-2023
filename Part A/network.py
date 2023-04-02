import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import gradient_descent_v2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std

H = 0.001  # Learning rate.
M = 0.0  # Momentum
NW_IN = 22  # Size of the input of the network.
NW_OUT = 5  # Output of the network (labels). Equal to the ammount of classes.
EPOCHS = 20  # Epochs in training
BATCH_SIZE = 20  # Batch size

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
      Dense(NW_OUT, input_dim=NW_IN, kernel_initializer='he_uniform',
            activation='relu'))
  model.add(Dense(NW_OUT, activation='sigmoid'))
  model.compile(loss='binary_crossentropy',
                optimizer=gradient_descent_v2.SGD(learning_rate=H, momentum=M),
                metrics=['binary_accuracy', 'MSE'])
  return model

def evaluateModel(X, y, model: Sequential) -> tuple[object, list]:
  results = list()
  cv = KFold(n_splits=5, shuffle=True)
  best_acc = 0
  for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = getModel()
    history = model.fit(X_train, y_train, verbose=1, epochs=EPOCHS,
                        validation_data=(X_test, y_test))
    predicted_y = model.predict(X_test)
    predicted_y = predicted_y.round()
    acc = accuracy_score(y_test, predicted_y)

    if acc > best_acc:
      best_acc = acc
      best_history = history

    # store result
    print('>%.3f' % acc)
    results.append(acc)

  # summarize performance
  summary = 'Accuracy: %.3f (%.3f)' % (mean(results), std(results))
  return [best_history, results, summary]

def plot_result(history, item):
  """Used for the graphs. Takes as input the name of the metric and 
    outputs graphs with the progression of said metric over EPOCHS."""

  plt.plot(history.history[item], label=item)
  plt.xlabel("Epochs")
  plt.ylabel(item)
  plt.title("Train {} Over Epochs".format(item), fontsize=14)
  plt.legend()
  plt.grid()
  plt.show()

def plot_history(history) -> None:
  items = ["loss", "MSE", "binary_accuracy"]
  items = ['val_' + item for item in items]

  [plot_result(history, item) for item in items]
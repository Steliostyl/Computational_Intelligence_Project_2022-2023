import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import gradient_descent_v2
from keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split

H = 0.005  # Learning rate.
M = 0.0  # Momentum
NW_IN = 22  # Size of the input of the network.
NW_OUT = 5  # Output of the network (labels). Equal to the ammount of classes.
EPOCHS = 100  # Epochs in training
BATCH_SIZE = 20  # Batch size

def getNetworkInput(processed_df: pd.DataFrame) -> list:
  """Accepts the processed dataset dataframe as input
  and returns a tuple of X, y used to train the model later."""

  class_categories = [
      'class_sitting', 'class_sittingdown', 'class_standing',
      'class_standingup', 'class_walking'
  ]

  X_train, X_test = train_test_split(
      processed_df.drop(class_categories, axis=1), test_size=0.3)
  y_train, y_test = train_test_split(processed_df[class_categories],
                                     test_size=0.3)

  return [
      X_train.values, X_test.values,
      y_train.values.astype(int),
      y_test.values.astype(int)
  ]

def getModel():
  model = Sequential()
  model.add(
      Dense(NW_IN + NW_OUT, input_dim=NW_IN, kernel_initializer='he_uniform',
            activation='relu'))
  model.add(Dense(NW_OUT, activation='sigmoid'))
  model.compile(
      loss="binary_crossentropy",
      #optimizer=gradient_descent_v2.SGD(learning_rate=H, momentum=M),
      optimizer='adam',
      metrics=["MSE", "binary_accuracy"])
  return model

callback_function = EarlyStopping(monitor="val_binary_accuracy", patience=10,
                                  min_delta=0.01, mode="max")

def evaluate_model(X_train, X_test, y_train, y_test) -> list:
  # Split the data to training and testing data 5-Fold
  kfold = KFold(n_splits=5, shuffle=True)

  model_results = []
  max_accuracy = 0

  for i, (train_index, test_index) in enumerate(kfold.split(X_train)):
    # Create model
    model = getModel()

    # Train model using KFOLD
    history = model.fit(
        X_train[train_index],  # Input for training
        y_train[train_index],  # Labels for training
        validation_data=(X_train[test_index],
                         y_train[test_index]),  # Validation data for KFOLD
        epochs=EPOCHS,
        batch_size=0,
        shuffle=True,
        callbacks=[callback_function])

    # Evaluate trained model and save the desired metrics (defined in model.compile)
    loss, mse, binary_acc = model.evaluate(X_train[test_index],
                                           y_train[test_index])
    model_results.append((loss, mse, binary_acc))

    # If current model has the best accuracy, save it as the best model
    if binary_acc > max_accuracy:
      best_model = model
      best_history = history

  # Print results for each test (fold)
  for i, result in enumerate(model_results):
    print(result[3])
    print(f"Fold number {i+1}. Test accuracy: {round(result[3] * 100, 2)}%.")

  # Evaluate the best model derived from previous KFOLD training on our real test data
  train_loss, train_mse, train_acc, train_bin_acc = best_model.evaluate(
      X_train, y_train, verbose=0)
  test_loss, test_mse, test_acc, test_bin_acc = best_model.evaluate(
      X_test, y_test, verbose=0)

  # Print the validation results of the best model
  print(
      f"Train accuracy: {round(train_acc * 100, 2)}%, Test accuracy: {round(test_acc * 100, 2)}%"
  )
  print(
      f"Train binary accuracy: {round(train_bin_acc * 100, 2)}%, Test binary accuracy: {round(test_bin_acc * 100, 2)}%"
  )
  print("Train MSE:", train_mse, "Test MSE:", test_mse)
  print("Train loss:", train_loss, "Test loss:", test_loss)

  return [model, history]

def plot_result(history, item):
  """Used for the graphs. Takes as input the name of the metric and 
    outputs graphs with the progression of said metric over EPOCHS."""

  plt.plot(history.history[item], label=item)
  plt.xlabel("Epochs")
  plt.ylabel(item)
  plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
  plt.legend()
  plt.grid()
  plt.show()

def plot_history(history) -> None:
  items = ["loss", "binary_accuracy", "mse"]

  [plot_result(history, item) for item in items]
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import matplotlib.pyplot as plt

def getNetworkInput(processed_df: pd.DataFrame) -> tuple[Sequential, object]:
    """Accepts the processed dataset dataframe as input
    and returns a tuple of X, y used to train the model later."""

    class_categories = ['class_sitting', 'class_sittingdown',
                                  'class_standing', 'class_standingup',
                                  'class_walking']
    X = processed_df.drop(class_categories + ['label'], axis=1).values
    y = processed_df[class_categories].values.astype(int)
    
    return (X, y)

def trainNetwork(X, y) -> tuple:
    """Trains a DNN model to predict user's stance based
    on input data (sensor data and personal information)"""

    print(X.shape)
    print(y.shape)

    # Define model
    model = Sequential()
    model.add(Dense(256, input_dim=X.shape[1], activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')
    # Train model
    history = model.fit(X, y, verbose=1, batch_size=10, epochs=3, validation_split=0.7)
    return (model, history)

def plot_history(history) -> None:
    items = ["loss", "binary_accuracy"]
    print(history.history)

    for item in items:
        plt.plot(history.history[item], label=item)
        plt.plot(history.history["val_" + item], label="val_" + item)
        plt.xlabel("Epochs")
        plt.ylabel(item)
        plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()
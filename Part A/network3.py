import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from sklearn.metrics import hamming_loss

# Define the neural network architecture
def create_model():
  model = Sequential()
  model.add(Dense(27, input_dim=22, activation='relu'))
  model.add(Dense(5, activation='sigmoid'))
  return model

def train_network(X, y):
  # Set up the optimizer
  sgd = SGD(learning_rate=0.001, momentum=0.6)

  # Create a StratifiedKFold object to split the data into 5 folds
  kfold = KFold(n_splits=5, shuffle=True)

  # Initialize an array to store the hamming loss for each fold
  hamming_loss_array = []

  # Loop over the folds
  for train, test in kfold.split(X, y):
    # Create a new neural network model for each fold
    model = create_model()

    # Compile the model with the specified optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=['accuracy', 'binary_accuracy', 'MSE'])

    # Train the model on the training data for 50 epochs
    model.fit(X[train], y[train], epochs=5, batch_size=32, verbose=1)

    # Evaluate the model on the test data
    y_pred = model.predict(X[test])
    hamming_loss_fold = hamming_loss(y[test], y_pred.round())
    hamming_loss_array.append(hamming_loss_fold)
    print(f'Hamming loss: {hamming_loss_fold:.2f}')

  # Print the average hamming loss over all folds
  print(f'Average hamming loss: {np.mean(hamming_loss_array):.2f}')

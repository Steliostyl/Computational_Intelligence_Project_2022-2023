from keras.models import Sequential
from keras.layers import Dense, Dropout

def get_model():
  # Define the neural network architecture
  model = Sequential()
  model.add(Dense(64, input_shape=(13, ), activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(5, activation='softmax'))

  # Compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['binary_accuracy', 'categorical_accuracy', 'MSE'])
  # Print the summary of the model
  model.summary()

  return model

def train_model(X, y):
  model = get_model()
  model.fit(X, y, epochs=10, batch_size=8, validation_split=0.3, verbose=1,
            shuffle=True)

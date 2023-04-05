import data_prep
import network
import network3
from keras.models import load_model, save_model
import pickle
from pprint import pprint

def main():
  preprocessed_df = data_prep.preprocessDataset()

  X, y = network.getNetworkInput(preprocessed_df)
  best_history, final_metrics = network.evaluateModel(X, y)

  pprint(final_metrics)
  network.plot_history(best_history)

  #history = network.train_test_model(X, y)
  #network.plot_history(history)

  #network3.train_network(X, y)

if __name__ == '__main__':
  main()
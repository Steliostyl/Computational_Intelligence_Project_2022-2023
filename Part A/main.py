import data_prep
import network
import network
from keras.models import load_model, save_model
import pickle
from pprint import pprint

def main():
  preprocessed_df = data_prep.preprocessDataset()

  X, y = network.getNetworkInput(preprocessed_df)
  best_history, final_metrics = network.evaluateModel(X, y)

  #save_model(model, "Part A/Files/model")
  #pprint(final_metrics)
  #network.plot_history(best_history)
  # model = load_model('Part A/model')

if __name__ == '__main__':
  main()
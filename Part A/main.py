import data_prep
import network

def main():
  preprocessed_df = data_prep.preprocessDataset()

  X, y = network.getNetworkInput(preprocessed_df)
  best_history, final_metrics = network.evaluateModel(X, y)

  print(final_metrics)
  network.plot_history(best_history)

if __name__ == '__main__':
  main()
import data_prep
import network

def main():
    preprocessed_df = data_prep.preprocessDataset()
    #print(preprocessed_df)

    X, y = network.getNetworkInput(preprocessed_df)
    model, history = network.trainNetwork(X, y)
    network.plot_history(history)
    

if __name__ == '__main__':
    main()
import data_prep

def main():
    preprocessed_df = data_prep.preprocessDataset()
    print(preprocessed_df)
    return

if __name__ == '__main__':
    main()
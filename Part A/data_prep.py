from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def plotHistogram(values, name) -> None:
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=values, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(name)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

def plot2(values, name):
    plt.subplots(figsize=(8, 2))
    plt.boxplot(values, vert=False)
    plt.ylabel("Features")
    plt.title(name)
    plt.show()

def encodeCatFeatures(original_df: pd.DataFrame) -> pd.DataFrame:
    return

def preprocessDataset() -> pd.DataFrame:
    # Load dataset from file
    files_folder = Path("Part A/Files/")
    original_df = pd.read_csv(files_folder / "dataset-HAR-PUC-Rio.csv", delimiter=';', low_memory=False, decimal=',')

    # Save column order
    original_columns = original_df.columns

    # Extract numerical columns
    numerical_columns = original_df.select_dtypes(include="number")

    # Normalize (min max scale) numerical values
    normalized_values = MinMaxScaler().fit_transform(numerical_columns.values)
    normalized_df = pd.DataFrame(columns=numerical_columns.columns, data=normalized_values)

    # Print initial and normalized dataframe head
    print(numerical_columns.head())
    print(normalized_df.head())

    # One-hot encode categorical features
    encoded_cat_features = encodeCatFeatures(original_df)

    return


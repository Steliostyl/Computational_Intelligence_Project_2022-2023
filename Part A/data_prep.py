import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

CLASSES = [
    "class_sitting", "class_sittingdown", "class_standing", "class_standingup",
    "class_walking"
]

def plotHistogram(values, name) -> None:
  # An "interface" to matplotlib.axes.Axes.hist() method
  n, bins, patches = plt.hist(x=values, bins='auto', color='#0504aa', alpha=0.7,
                              rwidth=0.85)
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

def preprocessDataset() -> pd.DataFrame:
  # Read dataset file and load it to a dataframe
  original_df = pd.read_csv("dataset-HAR-PUC-Rio.csv", delimiter=';',
                            low_memory=False, decimal=',')

  # Extract numerical columns
  numerical_columns = original_df.select_dtypes(include="number")

  # Testing different feature selection
  #numerical_columns = original_df[[
  #"how_tall_in_meters",
  #    "x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3",
  #    "z3", "x4", "y4", "z4"
  #]]

  # Standardize numerical values
  normalized_values = StandardScaler().fit_transform(numerical_columns.values)
  normalized_df = pd.DataFrame(columns=numerical_columns.columns,
                               data=normalized_values)

  # One-hot encode categorical features
  encoded_cat_features = pd.get_dummies(
      original_df.drop(numerical_columns.columns, axis=1))

  # Combine the 2 dataframes
  final_df = pd.concat([normalized_df, encoded_cat_features], axis=1)

  # Save processed dataset to file
  final_df.to_csv("Processed dataset.csv", index=False)

  return final_df
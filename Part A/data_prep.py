from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce

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

def oneHotEncodeColumns(categ_cols_df: pd.DataFrame) -> pd.DataFrame:
    """One hot encodes the columns of categorical features (columns)
    of the dataset (dataframe)."""

    # One hot encode categorical columns (that have no order)
    encoder=ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
    categ_cols_df = encoder.fit_transform(categ_cols_df)
    return categ_cols_df

def preprocessDataset() -> pd.DataFrame:
    # Load dataset from file
    files_folder = Path("Part A/Files/")
    original_df = pd.read_csv(files_folder / "dataset-HAR-PUC-Rio.csv", delimiter=';', low_memory=False, decimal=',')

    # Extract numerical columns
    numerical_columns = original_df.select_dtypes(include="number")

    # Normalize (min max scale) numerical values
    normalized_values = MinMaxScaler().fit_transform(numerical_columns.values)
    normalized_df = pd.DataFrame(columns=numerical_columns.columns, data=normalized_values)

    # One-hot encode categorical features
    encoded_cat_features = oneHotEncodeColumns(original_df[['user', 'class', 'gender']])

    # Combine the 2 dataframes
    final_df = pd.concat([normalized_df, encoded_cat_features], axis=1)

    # Reorder columns
    cols = final_df.columns.to_list()
    final_df.columns = cols[:-7] + ["gender_man", "gender_woman"] + cols[-7:-2]
    
    final_df['label'] = final_df[['class_sitting', 'class_sittingdown',
                                  'class_standing', 'class_standingup',
                                  'class_walking']].values.tolist()
    

    return final_df
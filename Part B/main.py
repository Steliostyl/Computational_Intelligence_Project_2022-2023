import pandas as pd
import ga
from functions import SENSOR_LIST
import functions


def main() -> None:
    dataset = pd.read_csv(
        "Part A/Files/dataset-HAR-PUC-Rio.csv",
        delimiter=";",
        low_memory=False,
        decimal=",",
    )[SENSOR_LIST + ["class"]]
    scaler = functions.getStandardScaler(dataset[SENSOR_LIST])
    normalized_dataset = dataset.copy(deep=True)
    normalized_dataset[SENSOR_LIST] = scaler.transform(
        normalized_dataset[SENSOR_LIST].values
    )
    # print(normalized_dataset)

    class_stats_df = functions.getClassStats(normalized_dataset)
    # print(class_stats_df)

    random_ind = ga.generateRandomIndividual(class_stats_df)
    random_ind.update_fitness_score(class_stats_df)
    print(f"Fitness score of individual: {random_ind.fitness_score}")


if __name__ == "__main__":
    main()

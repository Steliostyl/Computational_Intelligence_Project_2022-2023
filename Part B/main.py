import pandas as pd
import ga
from functions import SENSOR_LIST
import functions
from sklearn.preprocessing import StandardScaler

GENERATIONS = 50


def main() -> None:
    dataset = pd.read_csv(
        "Part A/Files/dataset-HAR-PUC-Rio.csv",
        delimiter=";",
        low_memory=False,
        decimal=",",
    )[SENSOR_LIST + ["class"]]
    scaler = StandardScaler().fit(dataset[SENSOR_LIST].values)
    normalized_dataset = dataset.copy(deep=True)
    normalized_dataset[SENSOR_LIST] = scaler.transform(
        normalized_dataset[SENSOR_LIST].values
    )
    # print(normalized_dataset)

    class_stats_df = functions.getClassStats(normalized_dataset)
    # print(class_stats_df)

    new_population = ga.spawnRandomPopulation(10, class_stats_df, 0)
    populations = [new_population]

    for i in range(GENERATIONS - 1):
        new_population = ga.spawnNewPopulation(new_population, class_stats_df, i)
        populations.append(new_population)

    avg_sensor_values = class_stats_df.loc[
        (class_stats_df["class"] == "sitting") & (class_stats_df["metric"] == "Average")
    ][SENSOR_LIST].values[0]
    dataset_fscore = functions.calculateFScore(avg_sensor_values, class_stats_df)
    functions.plotGenerations(populations, dataset_fscore)
    print(populations[-1].best_ind.fitness_score)
    print(populations[-1].best_ind.sensor_data)
    print(dataset_fscore)
    print(avg_sensor_values)


if __name__ == "__main__":
    main()

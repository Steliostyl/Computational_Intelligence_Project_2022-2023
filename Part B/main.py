import pandas as pd
import ga
from functions import SENSOR_LIST
import functions
from sklearn.preprocessing import StandardScaler


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

    random_ind = ga.spawnRandomIndividual(class_stats_df)
    random_ind.updateFitnessScore(class_stats_df)
    print(f"Fitness score of individual: {random_ind.fitness_score}")

    new_population = ga.spawnRandomPopulation(10, class_stats_df)
    new_population.calculateFitnessScores()
    parents = new_population.biasedWheelSelection()
    print([x.fitness_score for x in parents])


if __name__ == "__main__":
    main()

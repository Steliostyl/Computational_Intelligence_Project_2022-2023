import pandas as pd
import ga
from ga import SENSOR_LIST, INIT_MP, CP
from sklearn.preprocessing import StandardScaler

MAX_GENS = 1000  # Maximum generations (Only reached if algorithm keeps improving)
POP_SIZE = 20  # Population size
MIN_IMPROVEMENT = 0.01  # Minimum improvement of fitness function to not terminate
MAX_GENS_WO_IMPR = 20  # Maximum generations without improvement before terminating
MP = 0.00  # Mutation probability


def benchmark(runs: int) -> None:
    # Read the dataset
    dataset = pd.read_csv(
        "Part A/Files/dataset-HAR-PUC-Rio.csv",
        delimiter=";",
        low_memory=False,
        decimal=",",
    )[SENSOR_LIST + ["class"]]

    # Standardize values of dataset and create a new standardized dataset
    standardized_values = StandardScaler().fit_transform(dataset[SENSOR_LIST].values)
    standardized_dataset = dataset.copy(deep=True)
    standardized_dataset[SENSOR_LIST] = standardized_values

    # Create stats for each class (Min, Max and Average per-sensor values)
    class_stats_df = ga.getClassStats(standardized_dataset)

    total_generations = 0
    total_fitness = 0
    best_fitness = 0

    for _ in range(runs):
        # Spawn the first population randomly
        new_population = ga.spawnRandomPopulation(POP_SIZE, class_stats_df)
        populations = [new_population]
        # new_population.printPopulation()

        # Initialize mutation propability and decrease it over each generation
        mp = INIT_MP
        if INIT_MP > 0.01:
            decrease_rate = (INIT_MP - 0.01) / MAX_GENS
        else:
            decrease_rate = 0

        last_fitness = new_population.individuals[-1].fitness_score
        gens_since_last_improvement = 0

        # Main loop
        for i in range(1, MAX_GENS):
            # Decrease mutation rate
            # mp -= decrease_rate
            # Spawn a new generation
            new_population = ga.spawnNewPopulation(
                new_population, class_stats_df, i, MP
            )
            # Append new generation to the "Populations" list
            populations.append(new_population)
            # new_population.printPopulation()

            # Early termination if no improvement is seen for a while
            if (
                new_population.individuals[-1].fitness_score - last_fitness
            ) / last_fitness > MIN_IMPROVEMENT:
                last_fitness = new_population.individuals[-1].fitness_score
                gens_since_last_improvement = 0
            elif gens_since_last_improvement == MAX_GENS_WO_IMPR:
                break
            gens_since_last_improvement += 1

        # Calculate the fitness score of the dataset's average sensor values for sitting
        avg_sensor_values = class_stats_df.loc[
            (class_stats_df["class"] == "sitting")
            & (class_stats_df["metric"] == "Average")
        ][SENSOR_LIST].values[0]
        dataset_fscore = ga.calculateFScore2(avg_sensor_values, class_stats_df)
        total_generations += i
        total_fitness += populations[-1].individuals[-1].fitness_score
        if populations[-1].individuals[-1].fitness_score > best_fitness:
            best_fitness = populations[-1].individuals[-1].fitness_score
            populations_from_best_run = populations

    ga.plotGenerations(populations_from_best_run, dataset_fscore)

    print(f"Average number of generations: {total_generations/runs}")
    print(f"Average best fitness score: {total_fitness/runs}")

    ## Plot generational average, as well as best fitness scores
    # ga.plotGenerations(populations, dataset_fscore)
    #
    ## Print final fitness scores and sensor values
    # print(
    #    "Best individual from final population:\nFitness score: ",
    #    populations[-1].individuals[-1].fitness_score,
    # )
    # print("Sensor values: ", populations[-1].individuals[-1].sensor_data)
    # print("Fitness score of average sensor values from dataset: ", dataset_fscore)
    # print("Sensor values: ", avg_sensor_values)


benchmark(10)

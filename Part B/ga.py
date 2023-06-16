import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from scipy.spatial.distance import cosine

SITTING_WEIGHT = 2
CP = 0.6  # Crossover probability
INIT_MP = 0.2  # Initial mutation probability
C = 1

SENSOR_LIST = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "x4", "y4", "z4"]


def calculateFScore2(sensor_data: list[float], class_stats_df: pd.DataFrame) -> float:
    from_sitting = cosine(
        sensor_data,
        class_stats_df.loc[
            (class_stats_df["class"] == "sitting")
            & (class_stats_df["metric"] == "Average")
        ][SENSOR_LIST].values[0],
    )

    from_the_rest = -from_sitting
    for cl in class_stats_df["class"].unique():
        from_the_rest += cosine(
            sensor_data,
            class_stats_df.loc[
                (class_stats_df["class"] == cl)
                & (class_stats_df["metric"] == "Average")
            ][SENSOR_LIST].values[0],
        )
    fitness_score = ((from_sitting + C) * (1 - (from_the_rest / 4))) / 1 + C

    return fitness_score


def calculateFScore(sensor_data: list[float], class_stats_df: pd.DataFrame) -> float:
    # Calculate the euclidean distance of the person's sensor
    # values and the average sitting sensor values
    distance_from_sitting = np.linalg.norm(
        sensor_data
        - class_stats_df.loc[
            (class_stats_df["class"] == "sitting")
            & (class_stats_df["metric"] == "Average")
        ][SENSOR_LIST].values
    )

    if distance_from_sitting != 0:
        distance_from_sitting = max(distance_from_sitting, 0.5)
    else:
        distance_from_sitting = 0.5
    bonus_from_sitting = SITTING_WEIGHT / distance_from_sitting

    # Calculate the sum of the euclidean distances of the
    # person's sensor values and the average sensor values
    # of each class
    distance_from_rest = -distance_from_sitting
    for cl in class_stats_df["class"].unique():
        distance_from_rest += np.linalg.norm(
            sensor_data
            - class_stats_df.loc[
                (class_stats_df["class"] == cl)
                & (class_stats_df["metric"] == "Average")
            ][SENSOR_LIST].values
        )
    bonus_from_rest = 4 / distance_from_rest
    fitness_score = bonus_from_sitting + bonus_from_rest

    return fitness_score


class Individual:
    """Defines an individual of the genetic algorithm."""

    def __init__(self, sensor_data: list[float]) -> None:
        self.sensor_data = sensor_data
        self.fitness_score = 0
        self.pr_n = 0

    def randomResetMutation(self, class_stats_df: pd.DataFrame, mp) -> None:
        """Performs random reset mutation to genes
        that are selected with probability MP."""

        for idx, _ in enumerate(self.sensor_data):
            if np.random.uniform(0, 1) > mp:
                continue
            min_sensor_value = class_stats_df.loc[
                (class_stats_df["class"] == "sitting")
                & (class_stats_df["metric"] == "Min")
            ].values[0]
            max_sensor_value = class_stats_df.loc[
                (class_stats_df["class"] == "sitting")
                & (class_stats_df["metric"] == "Max")
            ].values[0]
            self.sensor_data[idx] = np.random.uniform(
                min_sensor_value[idx], max_sensor_value[idx]
            )


class Population:
    def __init__(
        self, individuals: list[Individual], avg_fitness: float, id: int
    ) -> None:
        self.id = id
        self.individuals = individuals
        self.avg_fitness = avg_fitness

    def biasedWheelSelection(self) -> list[Individual, Individual]:
        """Chooses 2 individuals from current generation
        by performing biased wheel selection and returns them."""

        parent_1 = None
        parent_2 = None

        while (parent_2 is None) or (parent_1 == parent_2):
            number = np.random.uniform(0, 1)
            # print("Random number: ", number)
            for ind in self.individuals:
                if ind.pr_n > number:
                    selected_ind = ind
                    if parent_1 is None:
                        parent_1 = selected_ind
                    else:
                        parent_2 = selected_ind
                    break

        return [parent_1, parent_2]

    def printPopulation(self) -> None:
        print(f"Average fitness score of population {self.id}: {self.avg_fitness}")
        print(f"Best fitness score: {self.individuals[-1].fitness_score}")
        [print(ind.fitness_score) for ind in self.individuals]


def spawnRandomPopulation(pop_size: int, class_stats_df: pd.DataFrame):
    individuals = [spawnRandomIndividual(class_stats_df) for _ in range(pop_size)]
    fitness_sum = 0
    best_ind = individuals[0]
    for ind in individuals:
        # Add individual's fitness score to the sum
        fitness_sum += ind.fitness_score
        # Get elite individual of population
        if ind.fitness_score > best_ind.fitness_score:
            best_ind = ind

    # Calculate population's average fitness
    avg_pop_fitness = fitness_sum / len(individuals)

    # Sort individuals by fitness and assign them pr_n (used for roulette wheel selection)
    individuals = sorted(individuals, key=lambda x: x.fitness_score, reverse=False)
    prev_total = 0
    for ind in individuals:
        prev_total += ind.fitness_score / fitness_sum
        ind.pr_n = prev_total

    # Create a new population object and return it
    return Population(individuals, avg_pop_fitness, 0)


def spawnRandomIndividual(class_stats_df: pd.DataFrame) -> Individual:
    """Spawns and returns a random individual with sensor values in
    range of the dataset's mins and maxes for each sensor value."""

    # Get sitting stats from class stats
    sitting_stats = class_stats_df.copy(deep=True).loc[
        class_stats_df["class"] == "sitting"
    ]
    # Get minimum sensor values for the sitting class
    min_values = (
        sitting_stats.loc[sitting_stats["metric"] == "Min"]
        .drop(["class", "metric"], axis=1)
        .values[0]
    )
    # Get maximum sensor values for the sitting class
    max_values = (
        sitting_stats.loc[sitting_stats["metric"] == "Max"]
        .drop(["class", "metric"], axis=1)
        .values[0]
    )
    # Randomly generate sensor values in accepted ranges.
    values = [
        np.random.uniform(low=min_values[i], high=max_values[i])
        for i in range(len(min_values))
    ]

    # Create an individual, calculate their fitness score and return it
    new_individual = Individual(sensor_data=values)
    new_individual.fitness_score = calculateFScore2(values, class_stats_df)
    return new_individual


def singlePointcrossover(parents: list[Individual]) -> list[Individual]:
    """Performs random single point crossover
    on 2 parents with probability CP."""

    # Check whether parents are going to reproduce offsprings
    # with probability CP. If not, return the parents unchanged.
    if np.random.uniform(0, 1) > CP:
        return [copy.deepcopy(p) for p in parents]

    # Pick a random crossover point
    k = random.randrange(1, (len(parents[0].sensor_data) - 1))

    # Keep the first k values of 1st parent and
    # the rest values from the 2nd parent
    sensor_data_1 = parents[0].sensor_data[:k] + parents[1].sensor_data[k:]
    sensor_data_2 = parents[0].sensor_data[k:] + parents[1].sensor_data[:k]

    # Return the 2 offsprings
    return [Individual(sensor_data_1), Individual(sensor_data_2)]


def spawnNewPopulation(
    current_pop: Population, class_stats_df: pd.DataFrame, id: int, mp: float
) -> Population:
    individuals = []
    pop_size = len(current_pop.individuals)

    # Perform single point crossover on parents selected with wheel selection
    for i in range(0, pop_size // 2):
        individuals += singlePointcrossover(current_pop.biasedWheelSelection())

    # Initialize variables to be changed within the next loop
    best_ind = current_pop.individuals[-1]
    worst_fitness = best_ind.fitness_score
    fitness_sum = best_ind.fitness_score
    for i, ind in enumerate(individuals):
        # Mutate individuals
        ind.randomResetMutation(class_stats_df, mp)
        # Calculate fitness scores for new individuals
        ind.fitness_score = calculateFScore2(ind.sensor_data, class_stats_df)
        fitness_sum += ind.fitness_score
        # Save least fit individual
        if ind.fitness_score < worst_fitness:
            worst_fitness = ind.fitness_score
            worst_idx = i

        # Save most fit individual (new elite)
        elif ind.fitness_score > best_ind.fitness_score:
            # print(
            #    f"New elite found in population {id} with fitness score: {ind.fitness_score}"
            # )
            best_ind = ind

    # We need 1 slot for the elite of the previous gen
    if len(individuals) == pop_size:
        # Delete least fit individual and update fitness_sum accordingly
        fitness_sum -= individuals[worst_idx].fitness_score
        del individuals[worst_idx]

    # Keep best individual from previous population unchanged
    individuals.append(current_pop.individuals[-1])

    # Calculate average population fitness
    avg_fitness = fitness_sum / pop_size

    # Sort individuals by fitness and assign them pr_n (used for roulette wheel selection)
    individuals = sorted(individuals, key=lambda x: x.fitness_score, reverse=False)
    prev_total = 0
    for ind in individuals:
        prev_total += ind.fitness_score / fitness_sum
        ind.pr_n = prev_total

    return Population(individuals, avg_fitness, id)


def getClassStats(df: pd.DataFrame) -> pd.DataFrame:
    """Returns stats (Min, Max, Avg sensor values) for each class."""

    classes = df["class"].unique()
    rows = []
    for cl in classes:
        class_data_df = df.loc[df["class"] == cl].drop("class", axis=1)
        rows.append(class_data_df.mean().to_list() + [cl, "Average"])
        rows.append(class_data_df.min().to_list() + [cl, "Min"])
        rows.append(class_data_df.max().to_list() + [cl, "Max"])

    return pd.DataFrame(rows, columns=SENSOR_LIST + ["class", "metric"], index=None)


def plotGenerations(populations: list, dataset_fscore: float):
    """Plots generational average and best fitness scores"""

    avg_fscore = []
    best_ind = []
    for pop in populations:
        avg_fscore.append(pop.avg_fitness)
        best_ind.append(pop.individuals[-1].fitness_score)

    # avg_fscore, best_ind = zip(
    #    *[(pop.avg_fitness, pop.best_ind) for pop in populations]
    # )

    x = range(len(avg_fscore))

    # Plot the populations
    fig, ax1 = plt.subplots()

    # Display fitness score of average sitting values from dataset as a line in the plot
    ax1.axhline(
        dataset_fscore,
        color="r",
        linestyle="--",
        label="Best score from dataset's average sensor values",
    )
    ax1.plot(x, avg_fscore, marker=".", label="Average fitness score")
    ax1.plot(x, best_ind, marker=".", label="Best individual")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness score")

    # Set the title
    plt.title("Population fitness by generation")

    # Add the legend
    ax1.legend()

    # Display the plot
    plt.show()

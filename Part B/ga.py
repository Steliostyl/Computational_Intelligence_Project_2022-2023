import pandas as pd
import numpy as np
import random

BONUS_FACTOR = 10
SITTING_WEIGHT = 1
MP = 0.05

SENSOR_LIST = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "x4", "y4", "z4"]


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
    bonus_from_sitting = SITTING_WEIGHT * BONUS_FACTOR / distance_from_sitting

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
    bonus_from_rest = 4 * BONUS_FACTOR / distance_from_rest
    fitness_score = bonus_from_sitting + bonus_from_rest

    return fitness_score


class Individual:
    """Defines an individual of the genetic algorithm."""

    def __init__(self, sensor_data: list[float]) -> None:
        # self.sensor_data = pd.DataFrame([sensor_data], columns=SENSOR_LIST)
        self.sensor_data = sensor_data
        self.fitness_score = 0
        self.pr_n = 0

    def randomResetMutation(self, class_stats_df: pd.DataFrame) -> None:
        """Performs random reset mutation to genes
        that are selected with probability MP."""

        for idx, _ in enumerate(self.sensor_data):
            n = np.random.uniform(0, 1)
            if n < MP:
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
        self, individuals: list[Individual], class_stats_df: pd.DataFrame, id: int
    ) -> None:
        self.id = id
        self.individuals = individuals
        self.min_score = 100
        self.max_score = 0
        self.fitness_sum = 0
        self.avg_fitness = 0
        self.best_ind = None
        self.calculateFitnessScores(class_stats_df)

    def calculateFitnessScores(self, class_stats_df: pd.DataFrame) -> None:
        """Calculates fitness scores for all individuals and saves min/max
        scores (to be used later in fitness scores normalization)"""

        for individual in self.individuals:
            individual.fitness_score = calculateFScore(
                individual.sensor_data, class_stats_df
            )
            if individual.fitness_score < self.min_score:
                self.min_score = individual.fitness_score
            if individual.fitness_score > self.max_score:
                self.max_score = individual.fitness_score
                self.best_ind = individual
            self.fitness_sum += individual.fitness_score

        self.avg_fitness = self.fitness_sum / len(self.individuals)
        self.individuals = sorted(
            self.individuals, key=lambda x: x.fitness_score, reverse=False
        )
        prev_total = 0
        for ind in self.individuals:
            prev_total += ind.fitness_score / self.fitness_sum
            ind.pr_n = prev_total

    def biasedWheelSelection(self) -> tuple[Individual, Individual]:
        """Chooses 2 individuals from current generation
        by performing biased wheel selection and returns them."""

        parent_1 = None
        parent_2 = None

        # [print([ind.fitness_score, ind.pr_n]) for ind in self.individuals]

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

        # print(parent_1.fitness_score, parent_2.fitness_score)
        return (parent_1, parent_2)

    def printPopulation(self) -> None:
        print(f"Average fitness score of population {self.id}: {self.avg_fitness}")
        [print(ind.fitness_score) for ind in self.individuals]


def singlePointcrossover(parents: list[Individual]) -> list[Individual]:
    # Pick a random crossover point
    # print(f"Sensor data length: {len(parents[0].sensor_data[0])}")
    # print(f"Sensor data:\n{parents[0].sensor_data}")
    k = random.randrange(1, (len(parents[0].sensor_data) - 1))
    # Keep the first k values of 1st parent and
    # the rest values from the 2nd parent
    sensor_data_1 = parents[0].sensor_data[:k] + parents[1].sensor_data[k:]
    sensor_data_2 = parents[0].sensor_data[k:] + parents[1].sensor_data[:k]
    # print("Sensor data 1: ", sensor_data_1, type(sensor_data_1))
    return [Individual(sensor_data_1), Individual(sensor_data_2)]


def spawnRandomPopulation(pop_size: int, class_stats_df: pd.DataFrame, id: int):
    individuals = [spawnRandomIndividual(class_stats_df) for i in range(pop_size)]
    new_pop = Population(individuals, class_stats_df, id)
    return new_pop


def spawnRandomIndividual(class_stats_df: pd.DataFrame) -> Individual:
    sitting_stats = class_stats_df.copy(deep=True).loc[
        class_stats_df["class"] == "sitting"
    ]
    min_values = (
        sitting_stats.loc[sitting_stats["metric"] == "Min"]
        .drop(["class", "metric"], axis=1)
        .values[0]
    )
    max_values = (
        sitting_stats.loc[sitting_stats["metric"] == "Max"]
        .drop(["class", "metric"], axis=1)
        .values[0]
    )
    values = [
        np.random.uniform(low=min_values[i], high=max_values[i])
        for i in range(len(min_values))
    ]

    return Individual(sensor_data=values)


def spawnNewPopulation(
    current_pop: Population, class_stats_df: pd.DataFrame, id: int
) -> Population:
    individuals = []
    for i in range(len(current_pop.individuals) // 2):
        individuals += singlePointcrossover(current_pop.biasedWheelSelection())

    new_population = Population(individuals, class_stats_df, id)
    return new_population

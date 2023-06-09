from functions import SENSOR_LIST
import pandas as pd
import numpy as np
import random

BONUS_FACTOR = 10
SITTING_WEIGHT = 2


class Individual:
    """Defines an individual of the genetic algorithm."""

    def __init__(self, sensor_data: list) -> None:
        self.sensor_data = pd.DataFrame([sensor_data], columns=SENSOR_LIST)

    def updateFitnessScore(self, class_stats_df) -> None:
        # Calculate the euclidean distance of the person's sensor
        # values and the average sitting sensor values
        distance_from_sitting = np.linalg.norm(
            self.sensor_data.values
            - class_stats_df.loc[
                (class_stats_df["class"] == "sitting")
                & (class_stats_df["metric"] == "Average")
            ][SENSOR_LIST].values
        )
        bonus_from_sitting = SITTING_WEIGHT * BONUS_FACTOR / distance_from_sitting
        # Calculate the sum of the euclidean distances of the
        # person's sensor values and the average sensor values
        # of each class
        distance_from_rest = -distance_from_sitting
        for cl in class_stats_df["class"].unique():
            distance_from_rest += np.linalg.norm(
                self.sensor_data.values
                - class_stats_df.loc[
                    (class_stats_df["class"] == cl)
                    & (class_stats_df["metric"] == "Average")
                ][SENSOR_LIST].values
            )
        bonus_from_rest = 4 * BONUS_FACTOR / distance_from_rest

        # print(f"Sitting sensor values:")
        # print(
        #    class_stats_df.loc[
        #        (class_stats_df["class"] == "sitting")
        #        & (class_stats_df["metric"] == "Average")
        #    ][SENSOR_LIST]
        # )
        # print(f"Individual sensor values:")
        # print(self.sensor_data)
        # print(f"Distance from sitting: {distance_from_sitting}")
        # print(f"Bonus from sitting: {bonus_from_sitting}")
        # print(f"Summed distance from the rest: {distance_from_rest}")
        # print(f"Bonus from rest: {bonus_from_rest}")

        self.fitness_score = bonus_from_sitting + bonus_from_rest

    def mutate(self):
        return


class Population:
    def __init__(
        self, individuals: list[Individual], class_stats_df: pd.DataFrame
    ) -> None:
        self.individuals = individuals
        self.class_stats_df = class_stats_df
        self.min_score = 100
        self.max_score = 0

    def calculateFitnessScores(self) -> None:
        """Calculates fitness scores for all individuals and saves min/max
        scores (to be used later in fitness scores normalization)"""

        for individual in self.individuals:
            individual.updateFitnessScore(self.class_stats_df)
            if individual.fitness_score < self.min_score:
                self.min_score = individual.fitness_score
            if individual.fitness_score > self.max_score:
                self.max_score = individual.fitness_score

    def biasedWheelSelection(self) -> list[Individual, Individual]:
        """Chooses 2 individuals from current generation
        by performing biased wheel selection and returns them."""

        sorted_list = sorted(
            self.individuals, key=lambda x: x.fitness_score, reverse=False
        )
        print([ind.fitness_score for ind in sorted_list])

        parent_1 = None
        parent_2 = None

        while (parent_1 is None) or (parent_2 is None):
            number = np.random.uniform(self.min_score, self.max_score)
            print("Random number: ", number)
            for ind in sorted_list:
                if ind.fitness_score < number:
                    continue
                selected_ind = ind
                if parent_1 is None:
                    parent_1 = selected_ind
                else:
                    parent_2 = selected_ind

        print(parent_1.fitness_score, parent_2.fitness_score)
        return [parent_1, parent_2]

    def printPopulation(self) -> None:
        print([ind.fitness_score for ind in self.individuals])


def crossover(first_parent: Individual, second_parent: Individual) -> Individual:
    offspring = None
    return offspring


def spawnRandomPopulation(pop_size: int, class_stats_df: pd.DataFrame):
    individuals = [spawnRandomIndividual(class_stats_df) for i in range(pop_size)]
    return Population(individuals, class_stats_df)


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

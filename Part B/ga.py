from functions import SENSOR_LIST
import pandas as pd
import numpy as np
import random

BONUS_FACTOR = 10


class Individual:
    """Defines an individual of the genetic algorithm."""

    def __init__(self, sensor_data: list, class_stats_df: pd.DataFrame) -> None:
        self.sensor_data = pd.DataFrame([sensor_data], columns=SENSOR_LIST)
        pass

    def update_fitness_score(self, class_stats_df) -> None:
        distance_from_sitting = np.linalg.norm(
            self.sensor_data.values
            - class_stats_df.loc[
                (class_stats_df["class"] == "sitting")
                & (class_stats_df["metric"] == "Average")
            ][SENSOR_LIST].values
        )
        bonus_from_sitting = BONUS_FACTOR / distance_from_sitting
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
        print(f"Sitting sensor values:")
        print(
            class_stats_df.loc[
                (class_stats_df["class"] == "sitting")
                & (class_stats_df["metric"] == "Average")
            ][SENSOR_LIST]
        )
        print(f"Individual sensor values:")
        print(self.sensor_data)
        print(f"Distance from sitting: {distance_from_sitting}")
        print(f"Bonus from sitting: {bonus_from_sitting}")
        print(f"Summed distance from the rest: {distance_from_rest}")
        print(f"Bonus from rest: {bonus_from_rest}")
        self.fitness_score = bonus_from_sitting + bonus_from_rest

    def mutate(self):
        return


class Population:
    def __init__(self, individuals: list[Individual]) -> None:
        self.individuals = individuals

    def get_best_individual(self) -> Individual:
        return


def crossover(first_parent: Individual, second_parent: Individual) -> Individual:
    offspring = None
    return offspring


def spawn_new_population():
    return


def generateRandomIndividual(class_stats_df: pd.DataFrame) -> Individual:
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

    return Individual(sensor_data=values, class_stats_df=class_stats_df)

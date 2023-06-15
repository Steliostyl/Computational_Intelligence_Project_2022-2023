import pandas as pd
import matplotlib.pyplot as plt

SENSOR_LIST = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "x4", "y4", "z4"]


def getClassStats(df: pd.DataFrame) -> pd.DataFrame:
    classes = df["class"].unique()

    rows = []
    for cl in classes:
        class_data_df = df.loc[df["class"] == cl].drop("class", axis=1)
        rows.append(class_data_df.mean().to_list() + [cl, "Average"])
        rows.append(class_data_df.min().to_list() + [cl, "Min"])
        rows.append(class_data_df.max().to_list() + [cl, "Max"])

    return pd.DataFrame(rows, columns=SENSOR_LIST + ["class", "metric"], index=None)


def plotGenerations(populations: list, dataset_fscore: float):
    # Calculate fitness score of average sitting values from dataset

    avg_fscore = []
    best_ind = []
    for pop in populations:
        avg_fscore.append(pop.avg_fitness)
        best_ind.append(pop.best_ind.fitness_score)

    # avg_fscore, best_ind = zip(
    #    *[(pop.avg_fitness, pop.best_ind) for pop in populations]
    # )

    x = range(len(avg_fscore))

    # Plot the populations
    fig, ax1 = plt.subplots()

    ax1.axhline(
        dataset_fscore,
        color="r",
        linestyle="--",
        label="Best score from dataset's average sensor values",
    )
    ax1.plot(x, avg_fscore, marker="o", label="Average fitness score")
    ax1.plot(x, best_ind, marker="o", label="Best individual")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness score")

    # Set the title
    plt.title("Population fitness by generation")

    # Add the legend
    ax1.legend()

    # Display the plot
    plt.show()

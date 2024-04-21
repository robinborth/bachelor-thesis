from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import load_prefiltered, save_plot


def generate_distribution_plot() -> None:
    sns.set_theme(style="whitegrid")
    prefiltered = load_prefiltered()
    triggers = prefiltered.explode(column="triggers")

    keyword_total_rank = triggers["triggers"].value_counts()

    blog = triggers.loc[triggers["source"] == "blog", "triggers"].value_counts()
    discord = triggers.loc[triggers["source"] == "discord", "triggers"].value_counts()
    docs = triggers.loc[triggers["source"] == "docs", "triggers"].value_counts()

    frequent_triggers: List[str] = list(triggers["triggers"].value_counts().keys())[:5]

    percentage_blog: List[float] = []
    percentage_discord: List[float] = []
    percentage_docs: List[float] = []

    for trigger in frequent_triggers:
        percentage_blog.append(blog[trigger] / blog.sum())
        percentage_discord.append(discord[trigger] / discord.sum())
        percentage_docs.append(docs[trigger] / docs.sum())

    percentage_blog.append(1 - sum(percentage_blog))
    percentage_discord.append(1 - sum(percentage_discord))
    percentage_docs.append(1 - sum(percentage_docs))
    frequent_triggers.append("others")

    percentage_blog = [i*100 for i in percentage_blog]
    percentage_discord = [i * 100 for i in percentage_discord]
    percentage_docs = [i * 100 for i in percentage_docs]

    X = np.arange(len(frequent_triggers))
    # Using X now to align the bars side by side
    plt.bar(X, percentage_blog, color="r", width=0.25)
    plt.bar(X + 0.25, percentage_discord, color="g", width=0.25)
    plt.bar(X + 0.5, percentage_docs, color="b", width=0.25)
    # Creating the legend of the bars in the plot
    plt.legend(["Discussion fora", "Discord chats", "User manuals"], loc="upper center")
    # Overiding the x axis with the country names
    plt.xticks([i + 0.25 for i in range(len(frequent_triggers))], frequent_triggers)
    plt.ylabel("Frequency (%)")
    # Saving the plot as a 'png'
    save_plot(plt, "causal_trigger_distribution.pdf")


if __name__ == "__main__":
    generate_distribution_plot()

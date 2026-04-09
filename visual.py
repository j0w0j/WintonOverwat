import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visual_barplot(df,category,outcome):
    plt.figure(figsize=(8,5))
    sns.barplot(x=category, y=outcome, data=df, estimator=lambda x: sum(x)/len(x))
    plt.ylabel("Percentage positieve prognose")
    plt.title(f"Invloed van {category} op prognose")
    os.makedirs("visuals", exist_ok=True)
    plt.savefig("visuals/visual_{}.png".format(category))
    plt.close()

def visual_other(df, variable, outcome):
    # Zorg dat de variabele numeriek is
    df = df.copy()
    df[variable] = pd.to_numeric(df[variable], errors="coerce")
    df = df.dropna(subset=[variable, outcome])

    plt.figure(figsize=(8,5))

    sns.regplot(
        x=variable,
        y=outcome,
        data=df,
        scatter_kws={'alpha':0.3},
        line_kws={'color': 'red'}
    )

    plt.ylabel("Percentage positieve prognose")
    plt.title(f"Relatie tussen {variable} en prognose")

    os.makedirs("visuals", exist_ok=True)
    plt.savefig(f"visuals/visual_{variable}.png")
    plt.close()



def main():
    df = pd.read_csv("data-studenten.csv")
    header = df.columns
    skip = ["Individu-ID","glucose", "cholesterol", "prognose10jaar", "Unnamed: 0"]

    for category in header:
        if category in skip:
            continue
        else:
            visual_barplot(df,category,"prognose10jaar")

    visual_other(df, "cholesterol", "prognose10jaar")
    visual_other(df, "glucose", "prognose10jaar")


if __name__ == "__main__": 
     main()
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

def visual_other(df,category,outcome):
    print("nee")
def main():
    df = pd.read_csv("data/data-studenten.csv")
    header = df.columns
    skip = ["Individu-ID","glucose", "cholesterol", "prognose10jaar", "Unnamed: 0"]

    for category in header:
        if category in skip:
            continue
        else:
            visual_barplot(df,category,"prognose10jaar")


if __name__ == "__main__": 
     main()
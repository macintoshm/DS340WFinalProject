import seaborn as sns
import pandas as pd

myTable = pd.read_csv("mergedTables.csv")



sns.set_theme(style="darkgrid")



myPlot = sns.countplot(x="label", data=myTable)

myPlot.set_xlabel("Label")
myPlot.set_ylabel("Count")
myPlot.set_title("Counts of Labels")

fig = myPlot.get_figure()
fig.savefig("labelsCountGraph.png")

print(myTable["label"].value_counts())
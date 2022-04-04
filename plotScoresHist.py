import matplotlib.pyplot as plt
import pandas as pd


myTable = pd.read_csv("mergedTables.csv")



plt.hist(myTable['score'])
plt.title("Sentiment Analysis Histogram")
plt.ylabel("Count")
plt.xlabel("Sentiment Score")

plt.savefig('plot.png', dpi=300, bbox_inches='tight')


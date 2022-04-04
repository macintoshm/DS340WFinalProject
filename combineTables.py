import pandas as pd


table1 = pd.read_csv("sentResults6000to12000.csv")
table2 = pd.read_csv("sentResults1to6000.csv")

myTable = table1.append(table2, ignore_index=True)

myTable.drop(myTable.columns[0], axis=1, inplace=True)

myTable.to_csv("mergedTables.csv")


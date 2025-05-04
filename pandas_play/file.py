import pandas as pd


def printdf(df):
    import sys

    sys.stdout.write(df.head().__str__() + "\n")


dataframe = pd.read_excel("finance.xlsx")
printdf(dataframe)


"""
feature engineer a new column by combining two columns together
"""
df = dataframe.copy()
df["newvalue"] = df.apply(lambda row: f"{row['Year']}-{row['Country']}", axis=1)
printdf(df)

"""
split and grab those two columns now.
"""
df_2 = pd.DataFrame(data=df["newvalue"])
df_2["Year"] = df_2.apply(lambda row: row["newvalue"].split("-")[0], axis=1)
df_2["Country"] = df_2.apply(lambda row: row["newvalue"].split("-")[1], axis=1)
df_2 = df_2.drop(columns=["newvalue"])
printdf(df_2)

print(len(df_2))

"""
filter out all the records where only those that have a country that start with the
letter G can stay.
"""
# -- or df_2 = df_2.query('Country == "Germany"')
df_2 = df_2[df_2["Country"].str.startswith("G")]
printdf(df_2)


""" TODO """

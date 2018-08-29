#%%
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("data/train.csv")
household_target = train_data[["Target", "idhogar"]]
household_target = household_target.drop_duplicates()

household_index = train_data.set_index(["idhogar", "Id"])
household_ids = train_data.idhogar.unique

household_features = [
    "r4h1",
    "r4t2",
    "r4h2",
    "r4h3",
    "r4m1",
    "r4m2",
    "r4m3",
    "r4t1",
    "r4t3",
    "area1",
    "area2",
    "bedrooms",
    "overcrowding",
    "meaneduc",
    "elimbasu1",
    "elimbasu2",
    "elimbasu3",
    "elimbasu4",
    "elimbasu5",
    "elimbasu6"
    ]

household_dataframe = household_index.fillna(0).groupby(level = 0).mean()
household_dataframe = household_dataframe.reset_index()

household_dataframe.to_pickle("data/household_features")

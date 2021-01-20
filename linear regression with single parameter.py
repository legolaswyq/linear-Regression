import pandas as pd
import matplotlib.pyplot as plt

filename = "ex1data2.txt"
data = pd.read_csv(filename)

print(data.head())
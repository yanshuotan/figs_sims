import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

from plot_util import plot_curves

with open("results/no_noise_results_linear.pkl", "rb") as file:
    results1 = pkl.load(file)

with open("results/high_dim_results_linear.pkl", "rb") as file:
    results2 = pkl.load(file)

H = 0.4
sim1 = [results for results in results1 if results["s"] == 6 and results["heritability"] == H][0]
sim2 = results2[0]
plot_curves(sim1, name=f"Noiseless Linear", title=False, export=True, with_linear=True)
plot_curves(sim2, name=f"High Dim Linear", title=False, export=True, with_linear=True)
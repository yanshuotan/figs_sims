import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

from plot_util import plot_curves

with open("results/linear_sim_results.pkl", "rb") as file:
    linear_results = pkl.load(file)

with open("results/lss_sim_results.pkl", "rb") as file:
    lss_results = pkl.load(file)

with open("results/poly_sim_results.pkl", "rb") as file:
    poly_results = pkl.load(file)

with open("results/hierarchical_lss_sim_results.pkl", "rb") as file:
    hierarchical_lss_results = pkl.load(file)

H = 0.4

sim1 = [results for results in linear_results if results["s"] == 6 and results["heritability"] == H][0]
sim2 = [results for results in poly_results if results["m"] == 1 and results["r"] == 3 and results["heritability"] == H][0]
sim3 = [results for results in poly_results if results["m"] == 2 and results["r"] == 2 and results["heritability"] == H][0]
sim4 = [results for results in lss_results if results["m"] == 5 and results["r"] == 2 and results["heritability"] == H][0]

sims = [sim1, sim2, sim3, sim4]
names = ["Linear", "Single poly", "Sum of polys", "LSS"]
for sim, name in zip(sims, names):
    plot_curves(sim, name=f"Fig 2 {name}", title=False, export=True, with_linear=True)
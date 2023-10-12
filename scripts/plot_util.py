import matplotlib.pyplot as plt
import matplotlib as mpl
import dvu


def plot_curves(results, with_linear=True, name=None, export=False):
    dvu.set_style()
    mpl.rcParams['figure.dpi'] = 250
    results_df = results["vals"]
    title_string = ""
    if name is not None:
        title_string += f"{name}, "
    for k, v in results.items():
        if k != "vals":
            title_string += f"{k} = {v}, "
    title_string = title_string[:-2]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_types = ["mse", "bias", "variance"]
    for idx, ax in enumerate(axes):
        plot_type = plot_types[idx]
        results_df_filtered = results_df[results_df["type"] == "figs"]
        ax.plot(results_df_filtered["n_rules"], results_df_filtered[plot_type], label='FIGS')
        results_df_filtered = results_df[results_df["type"] == "cart"]
        ax.plot(results_df_filtered["n_rules"], results_df_filtered[plot_type], label='CART')
        results_df_filtered = results_df[(results_df["type"] == "gb") & (results_df["max_depth"] == 1)]
        ax.plot(results_df_filtered["n_rules"], results_df_filtered[plot_type], label='GB (max depth=1)')
        results_df_filtered = results_df[(results_df["type"] == "gb") & (results_df["max_depth"] == 2)]
        ax.plot(results_df_filtered["n_rules"], results_df_filtered[plot_type], label='GB (max depth=2)')
        lin_model_val = results_df[results_df["type"] == "linear"][plot_type].values[0]
        if with_linear:
            ax.axhline(y=lin_model_val, label='Linear', ls='--', color="black")
        if plot_type == "mse":
            subtitle_string = "MSE"
        else:
            subtitle_string = plot_type.capitalize()
        ax.set_title(subtitle_string)
        ax.set_xlabel("No. of rules")
        ax.set_ylabel("Error")
        if idx == 2:
            ax.legend()
    plt.tight_layout()
    plt.suptitle(title_string, fontsize=16, y=1.1)
    if export:
        plt.savefig(f"plots/{name}_plot.pdf")
    plt.show()

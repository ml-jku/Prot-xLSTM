import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

from protxlstm.plot_utils import cd, setup_matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))

def interpolate(x,y, step, sigma=1):

    from scipy.ndimage import gaussian_filter

    x_smooth = np.arange(np.min(x), np.max(x), step)
    y_interp = np.interp(x_smooth, x, y)
    smoothed_y_interp = gaussian_filter(y_interp, sigma=sigma)

    return x_smooth, smoothed_y_interp



def plot(all_curves, metric='val_loss', step=-1, log = None):

    # Modelclass color mapping
    color_dict = {
        "Prot-xLSTM-26M": cd['xLSTM'],
        "ProtMamba-28M": cd['Mamba'],
        "Prot-Transformer++-26M": cd['Transformers'],
        "Prot-xLSTM-102M": cd['xLSTM'],
        "ProtMamba-107M" : cd['Mamba']
    }

    fig, axs = plt.subplots(1, 2, figsize=(6,2.2))

    if metric == 'val_loss':
        metric, y_low, y_high, y_label, save_name, protmamba_min = "eval/valid_loss/all", 1.5, 3.1, "Validation Loss", "val_loss", 1.93


    # Small models ----------------------------------------------------------------------------------------
    for experiment in ["Prot-xLSTM-26M", "ProtMamba-28M", "Prot-Transformer++-26M"]:
        x = np.array(all_curves[experiment][f'{metric}_x'])
        y = np.array(all_curves[experiment][f'{metric}_y'])
        if step != -1:
            x, y = interpolate(x,y,step)
        axs[0].plot(x, y, color=color_dict[experiment], linewidth=0.8, label = experiment)

    # Add transition lines
    axs[0].vlines(20, y_low, y_high, colors="grey", linestyles="--", linewidth=0.5)

    axs[0].set_xlabel('Billion Tokens') 
    axs[0].set_ylabel(y_label)
    axs[0].set_ylim([y_low,y_high])
    axs[0].grid("both")
    axs[0].legend(loc = 'lower left')

    # Large models -------------------------------------------------------------------------------------------

    x = np.array(all_curves["Prot-xLSTM-102M"][f'{metric}_x'])
    y = np.array(all_curves["Prot-xLSTM-102M"][f'{metric}_y'])
    if step != -1:
        x, y = interpolate(x,y,step)
    axs[1].plot(x, y, color=color_dict["Prot-xLSTM-102M"], linewidth=0.8, label = "Prot-xLSTM-102M")
    axs[1].hlines(protmamba_min, xmin=x[0], xmax=x[-1], color=color_dict["ProtMamba-107M"], linestyle="--", linewidth=0.8, label = "ProtMamba-107M")

    # Add transition lines
    transitions = all_curves['Prot-xLSTM-102M']['transitions']
    axs[1].vlines(transitions, y_low, y_high, colors="grey", linestyles="--", linewidth=0.5, zorder=0)

    axs[1].set_xlabel('Billion Tokens') 
    axs[1].set_ylim([y_low, y_high])
    axs[1].grid("both")
    axs[1].legend(loc='lower left')

    if log:
        formatter = FuncFormatter(lambda x, _: f'{x:.3g}')
        if log in ['both', 'small']:
            axs[0].set_xscale('log')
            axs[0].get_xaxis().set_major_formatter(formatter)
        if log in ['both', 'large']:
            axs[1].set_xscale('log')
            axs[1].get_xaxis().set_major_formatter(formatter)

    # axs[0].text(30, 2.07, r"$\downarrow$", color="k", ha='center', va='center', fontsize=5)
    axs[1].text(60, 1.85, r"$\downarrow$", color="k", ha='center', va='center', fontsize=5)


    fig.tight_layout()
    fig.savefig(os.path.join(HERE,f'ProtXLSTM_{save_name}.pdf'), dpi=1200, transparent=True)

if __name__ == "__main__":

    # Load curves
    with open(os.path.join(HERE,'learning_curves.json'), 'r') as f:
        all_curves = json.load(f)

    # Plot figure
    setup_matplotlib()
    plot(all_curves, step=0.05, log='large')
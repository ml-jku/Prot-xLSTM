
cd = { # use dependent on model-type!!
    "xLSTM": "#3073AD",
    "Transformers": "#4B9D7A",
    "Mamba": "#DF8953",
    "S4": "#D275AB",
    "Hyena": "#E86A61",
}

def setup_matplotlib():
  import matplotlib.pyplot as plt
  from tueplots import bundles, axes
  bundles.icml2022()
  plt.rcParams.update(bundles.icml2022())
  plt.rcParams.update(axes.lines(base_width=0.5))
  plt.rcParams["text.usetex"] = False
  plt.rcParams['font.family'] = "sans-serif"
  plt.rcParams['font.serif'] = 'Arial'
  plt.rcParams['legend.edgecolor'] = 'grey'
  plt.rcParams['legend.framealpha'] = 0.7
  plt.rcParams['lines.linewidth'] = 1.2
  plt.rcParams['axes.grid'] = True
  plt.rcParams['axes.grid.axis'] = 'both'
  plt.rcParams['grid.alpha'] = 0.2
  plt.rcParams['axes.grid'] = True
  plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cd.values())
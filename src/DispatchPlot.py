# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Author: dylanjm
  Date: 2021-05-18
"""
import itertools as it
import matplotlib as mpl
mpl.use('Agg') # Prevents the script from blocking while plotting
import matplotlib.pyplot as plt
from typing import List, Dict
import random
import numpy as np

try:
  from ravenframework.PluginBaseClasses.OutStreamPlotPlugin import PlotPlugin, InputTypes, InputData
except ModuleNotFoundError:
  import sys
  from . import _utils
  sys.path.append(_utils.get_raven_loc())
  from ravenframework.PluginBaseClasses.OutStreamPlotPlugin import PlotPlugin, InputTypes, InputData


# Matplotlib Global Settings
plt.rc("figure", figsize=(12, 8), titleweight='bold') # type: ignore
plt.rc("axes", titleweight="bold", labelsize=12, axisbelow=True, grid=True) # type: ignore
plt.rc("savefig", bbox="tight") # type: ignore
plt.rc("legend", fontsize=12) # type:ignore
plt.rc(["xtick", "ytick"], labelsize=10) # type: ignore


class DispatchPlot(PlotPlugin):

  @classmethod
  def getInputSpecification(cls):
    """
      Define the acceptable user inputs for this class.
      @ In, None
      @ Out, specs, InputData.ParameterInput,
    """
    specs = super().getInputSpecification()
    specs.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType))
    specs.addSub(InputData.parameterInputFactory('macro_variable', contentType=InputTypes.StringType))
    specs.addSub(InputData.parameterInputFactory('micro_variable', contentType=InputTypes.StringType))
    specs.addSub(InputData.parameterInputFactory('signals', contentType=InputTypes.StringListType))
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'HERON.DispatchPlot'
    self._sourceName = None
    self._source = None
    self._macroName = None
    self._microName = None
    self._addSignals = []

  def handleInput(self, spec):
    """
      Reads in data from the input file
      @ In, spec, InputData.ParameterInput, input information
      @ Out, None
    """
    super().handleInput(spec)
    for node in spec.subparts:
      if node.getName() == 'source':
        self._sourceName = node.value
      elif node.getName() == 'macro_variable':
        self._macroName = node.value
      elif node.getName() == 'micro_variable':
        self._microName = node.value
      elif node.getName() == 'signals':
        self._addSignals = node.value

  def initialize(self, stepEntities):
    """
      Set up plotter for each run
      @ In, stepEntities, dict, entities from the Step
      @ Out, None
    """
    super().initialize(stepEntities)
    src = self.findSource(self._sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'Source DataObject "{self._sourceName}" was not found in the Step!')
    self._source = src

  @staticmethod
  def _group_by(iterable: List[str], idx: int) -> Dict[str, List[str]]:
    """
      Return a dictionary containing grouped dispatch variables.
      @ In, iterable, List[str], a list of variable names to group-by.
      @ In, idx, int, the index of the variable to group-by.
      @ Out, gr, Dict[str, List[str]], a dictionary mapping of grouped variable names.
    """
    gr = {}
    for var in iterable:
      # var is expected to have the form: 'Dispatch__component__tracker__resource'
      key = var.split('__')[idx]
      if key in gr.keys():
        gr[key].append(var)
      else:
        gr[key] = [var]
    return gr

  def plot_component(self, fig, axes, df, grp_vars, comp_idx, sid, mstep, cid, cdict) -> None:
    """
      Plot and output the optimized dispatch for a specific sample, year, and cluster.
      @ In, fig, matplotlib.figure.Figure, current figure used for plotting.
      @ In, axes, List[List[matplotlib.Axes]], a list of axes to plot each variable.
      @ In, df, pandas.DataFrame, a dataframe containing data to plot.
      @ In, grp_vars, Dict[str, List[str]], a dictionary mapping components to variables.
      @ In, comp_idx, Dict[str, int], a dictionary mapping components to numbers.
      @ In, sid, int, the sample ID.
      @ In, mstep, int, the macro step.
      @ In, cid, int, the cluster ID.
      @ In, cdict, Dict[str, str], a dictionary contains color code to variables
      @ Out, None
    """
    # Pre-define color codes and transparency
    Gray, Dark = ('#dcddde','#1a2b3c')
    alpha = '70'
    for (key, group), ax in zip(grp_vars.items(), axes.flat):
      # Define list for data, label, and color. Seperate 'level'(line plot) with other variables (stack plot)
      positive_dat = []
      positive_label = []
      positive_color = []
      negative_dat = []
      negative_label = []
      negative_color = []
      level_dat = []
      level_label = []
      level_color = []

      # Secondary y axis for levels
      ax2 = ax.twinx()
      # Fill the lists
      for var in group:
        _, comp_name, tracker, _ = var.split('__')
        comp_label = comp_name.replace('_', ' ').title()
        var_label = f'{comp_label}, {tracker.title()}'
        ls = '-'
        # Fill the positive, negative, and level lists
        cindex = key + "," + comp_name # key for cdict dictionary
        if (df[var] != 0).any(): # no plotting variables that have all zeros values
          if tracker == 'level':
            level_dat.append(var)
            level_label.append(var_label)
            level_color.append(cdict.get(cindex))
          else:
            if (df[var] > 0).any():
                positive_dat.append(var)
                positive_label.append(var_label)
                positive_color.append(cdict.get(cindex))
            else:
                negative_dat.append(var)
                negative_label.append(var_label)
                negative_color.append(cdict.get(cindex))
      # Plot the micro-step variable on the x-axis (i.e Time)
      # Stackplot
      ax.stackplot(df[self._microName],*[df[key] for key in positive_dat],labels= positive_label, baseline='zero', colors= [color+alpha for color in positive_color[:len(negative_dat)]]+[Gray])
      ax.stackplot(df[self._microName],*[df[key] for key in negative_dat], labels= negative_label, baseline='zero', colors= [color+alpha for color in negative_color[:len(negative_dat)]] +[Gray])
      # Lineplot
      for key, c, llabel in zip(level_dat, level_color[:len(level_dat)] + [Dark], level_label[:len(level_dat)]):
        ax2.plot(df[self._microName], df[key], linestyle=ls, label=llabel, color=c )
      # Set figure title, legend, and grid
      ax.set_title(key.title().split('_')[-1])
      ax.set_xlabel(self._microName)
      if(len(positive_label) > 0 or len(negative_label) > 0):
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 0.6), fontsize = 10)
      if(len(level_label) > 0):
        ax2.legend(loc='lower left', bbox_to_anchor=(1.1, 0.6), fontsize = 10)
      # Add the label and adjust location
      ax.set_ylabel('Activity', fontsize=10, rotation=0)
      ax2.set_ylabel('Level', fontsize=10, rotation=0)
      ax.yaxis.set_label_coords(-0.01,1.02)
      ax2.yaxis.set_label_coords(1,1.07)
      ax.grid(None)
      ax2.grid(None)
    # Output and save the image
    file_name = f"dispatch_id{sid}_y{mstep}_c{cid}.png"
    fig.tight_layout()
    fig.savefig(file_name)
    self.raiseAMessage(f'Saved figure to "{file_name}"')
    plt.clf()

  def plot_signal(self, fig, axes, df, sid, mstep, cid) -> None:
    """
      Plot and output the synthetic history for a specific sample, year, and cluster.
      @ In, fig, matplotlib.figure.Figure, a current figure used for plotting.
      @ In, axes, List[List[matplotlib.Axes]], a list of axes to plot each variable.
      @ In, df, pandas.DataFrame, a dataframe containing data to plot.
      @ In, sid, int, the sample ID.
      @ In, mstep, int, the macro step.
      @ In, cid, int, the cluster ID.
    """
    for name, ax in zip(self._addSignals, axes.flat):
      var = df.get(name, None)
      if var is None:
        self.raiseAnError(f'Requested signal variable "{name}" but variable not in data!')
      ax.plot(df[self._microName], var, marker='.', linestyle='-', label=name)
      ax.set_title(name.title())
      ax.set_xlabel(self._microName)
      ax.legend(loc='center left', bbox_to_anchor=(1.03, 0.5))

    signal_file_name = f"dispatch_id{sid}_y{mstep}_c{cid}_SIGNAL.png"
    fig.tight_layout()
    fig.savefig(signal_file_name)
    self.raiseAMessage(f'Saved figure to "{signal_file_name}"')
    plt.clf()

  def color_style(self, grp_vars):
    """
      @ In, grp_vars, Dict[str, List[str]], a dictionary mapping components to variables.
      @ Out, colors, Dict[str, str], contains color code for variables
    """
    resources = [] # Determine the number of colormaps
    technologis = [] # Determine the number of colors obtained from a colormap
    for key, group in grp_vars.items():
      resources.append(key)
      for var in group:
        _, comp_name, tracker, _ = var.split('__')
        technologis.append(key + ',' + comp_name)
    # remve duplicates
    resources = list(dict.fromkeys(resources))
    technologis = list(dict.fromkeys(technologis))
    # colormap codes - can be changed to preferred colormaps - 17 in total 'Sequential' series
    cm_codes = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    sample_cm = random.sample(cm_codes, len(resources))
    resource_cm = {} # E.g. {'heat': 'OrRd', 'electricity': 'GnBu'} all string
    i = 0
    for s in resources:
      resource_cm[s] = sample_cm[i] + '_r' #reverse colormap so it won't pick the lightest color that is almost invisible
      i = i + 1
    # Get the number of colors needed
    resource_count = {} # E.g. {'heat': 5, 'electricity': 5}
    for s in resources:
      count = 0
      for t in technologis:
        if s in t:
          count = count + 1
      resource_count[s] = count
    # Assign colors
    colors = {}
    for s in resources:
      cm = mpl.cm.get_cmap(name= resource_cm[s])
      # Get a subset of color map from 0 - 0.8 to avoid invisble light colors
      cm = mpl.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cm.name, a=0, b=0.8),cm(np.linspace(0, 0.8)))
      j = 0
      for t in technologis:
        clist = [cm(1.*i/resource_count[s]) for i in range(resource_count[s])] #color list
        clist.reverse()
        if s in t:
          colors[t] = mpl.colors.rgb2hex(clist[j])
          j = j + 1
    return colors

  def run(self):
    """
      Generate the plot
      @ In, None
      @ Out, None
    """
    assert self._source is not None
    ds = self._source.asDataset()
    if ds is None:
      self.raiseAWarning(f'No data in "{self._source.name}" data object; nothing to plot!')
      return
    df = ds.to_dataframe().reset_index()
    dispatch_vars = list(filter(lambda x: "Dispatch__" in x, df.columns))
    grouped_vars = self._group_by(dispatch_vars, -1)
    grouped_comp = self._group_by(dispatch_vars, 1)
    comp_idx = {comp: i for i, comp in enumerate(grouped_comp.keys())}
    # Dimension variables to plot
    sample_ids = df[self._source.sampleTag].unique()
    cluster_ids = df['_ROM_Cluster'].unique()  # TODO: find way to not hardcode name
    macro_steps = df[self._macroName].unique()
    # Assign colors
    cdict = self.color_style(grouped_vars)
    for sample_id, macro_step, cluster_id in it.product(sample_ids, macro_steps, cluster_ids):
      # Filter data to plot correct values for current dimension
      dat = df[
        (df[self._source.sampleTag] == sample_id) &
        (df[self._macroName] == macro_step) &
        (df['_ROM_Cluster'] == cluster_id)
      ]

      # TODO: find a way to combine both plots into one output.
      # Currently, this is difficult to do because of the nested
      # nature of the subplots, as well as the dynamic number of
      # components and signals to plot (i.e. dynamically nested subplots)

      # Output optimized component dispatch for current dimension.
      fig0, axs0 = plt.subplots(len(grouped_vars), 1, sharex=True, squeeze=False)
      self.plot_component(fig0, axs0, dat, grouped_vars, comp_idx, sample_id, macro_step, cluster_id, cdict)

      # Output synthetic time series signal for current dimension.
      fig1, axs1 = plt.subplots(len(self._addSignals), 1, sharex=True, squeeze=False)
      self.plot_signal(fig1, axs1, dat, sample_id, macro_step, cluster_id)

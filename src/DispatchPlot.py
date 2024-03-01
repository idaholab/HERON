# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Author: dylanjm
  Date: 2021-05-18
"""
import itertools as it
from collections import defaultdict

import matplotlib as mpl
mpl.use('Agg') # Prevents the script from blocking while plotting
import matplotlib.pyplot as plt

from typing import List, Dict
import numpy as np

try:
  from ravenframework.PluginBaseClasses.OutStreamPlotPlugin import PlotPlugin, InputTypes, InputData
except ModuleNotFoundError:
  import sys
  from . import _utils
  sys.path.append(_utils.get_raven_loc())
  from ravenframework.PluginBaseClasses.OutStreamPlotPlugin import PlotPlugin, InputTypes, InputData


# default color cycler, hatches
colormap = plt.get_cmap('tab10').colors
hatchmap = [None, '..', '\\\\', 'xx', '--', 'oo', '++', '**', 'OO']

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

  def plot_dispatch(self, figs, axes, df, grp_vars, sid, mstep, cid, cdict) -> None:
    """
      Plot and output the optimized dispatch for a specific sample, year, and cluster.
      @ In, figs, matplotlib.figure.Figure, current figures used for plotting.
      @ In, axes, List[List[matplotlib.Axes]], a list of axes across figures to plot each variable.
      @ In, df, pandas.DataFrame, a dataframe containing data to plot.
      @ In, grp_vars, Dict[str, List[str]], a dictionary mapping components to variables.
      @ In, sid, int, the sample ID.
      @ In, mstep, int, the macro step.
      @ In, cid, int, the cluster ID.
      @ In, cdict, Dict[Dict[str, Tuple[Float]]], a dictionary contains color code to variables
      @ Out, None
    """
    alpha = 0.7
    time = df[self._microName].to_numpy()
    for (key, group), ax in zip(grp_vars.items(), axes.flat):
      # Define list for data, label, and color. Seperate 'level'(line plot) with other variables (stack plot)
      positive_dat = []
      positive_label = []
      positive_color = []
      positive_hatch = []

      negative_dat = []
      negative_label = []
      negative_color = []
      negative_hatch = []

      level_dat = []
      level_label = []
      level_color = []

      # Secondary y axis for storage levels
      ax2 = ax.twinx()

      # Fill the lists
      for var in group:
        _, comp_name, tracker, _ = var.split('__')
        comp_label = comp_name.replace('_', ' ')#.title()
        var_label = f'{comp_label}, {tracker.title()}'
        ls = '-'
        # Fill the positive, negative, and level lists
        info = cdict[comp_name]#[res][tracker]
        color = info['color']
        hatch = info['hatch']
        if (df[var] != 0).any(): # no plotting variables that have all zeros values
          if tracker == 'level':
            # this is the level of a storage component
            level_dat.append(var)
            level_label.append(var_label)
            level_color.append(color)
          else:
            if (df[var] > 0).any():
              # these are production tracking variables
              positive_dat.append(var)
              positive_label.append(var_label)
              positive_color.append(tuple([*color, alpha]))
              positive_hatch.append(hatch)
            else:
              # these are consumption tracking variables
              negative_dat.append(var)
              negative_label.append(var_label)
              negative_color.append(tuple([*color, alpha]))
              negative_hatch.append(hatch)

      # Plot the micro-step variable on the x-axis (i.e Time)
      # center (0) line
      ax.plot([time[0],time[-1]], [0,0], 'k-')
      # Production
      if len(positive_dat) > 0:
        pos_stacks = ax.stackplot(time,
                       *[df[key] for key in positive_dat],
                       labels=positive_label,
                       baseline='zero',
                       colors=positive_color)
        for stack, hatch in zip(pos_stacks, positive_hatch):
          stack.set_hatch(hatch)

      # Consumption
      if len(negative_dat) > 0:
        neg_stacks = ax.stackplot(time,
                       *[df[key] for key in negative_dat],
                       labels= negative_label,
                       baseline='zero',
                       colors=negative_color)
        for stack, hatch in zip(neg_stacks, negative_hatch):
          stack.set_hatch(hatch)

      # Levels
      if(len(level_dat) > 0):
        for key, c, llabel in zip(level_dat, level_color, level_label):
          ax2.plot(df[self._microName], df[key], linestyle=ls, label=llabel, color=c)

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
    for f, fig in enumerate(figs):
      file_name = f"dispatch_id{sid}_y{mstep}_c{cid}_f{f+1}.png"
      fig.suptitle(f'Dispatch ID {sid} Year {mstep} Cluster {cid},\nFigure {f+1}/{len(figs)}')
      fig.tight_layout()
      fig.savefig(file_name)
      self.raiseAMessage(f'Saved figure {f+1}/{len(figs)} to "{file_name}"')
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
        self.raiseAnError(RuntimeError, f'Requested signal variable "{name}" but variable not in data!')
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
      Set the coloring scheme for each of the variables that will be plotted.
      @ In, grp_vars, Dict[str, List[str]], a dictionary mapping components to variables.
      @ Out, comp_colors, Dict[Dict[str, Tuple[Float]]], contains rgb color code and hatching for components
    """
    # DESIGN:
    #   -> components should be clearly different in color, and consistent across resource plots
    # get the components, resources they use, and trackers per resource
    # TODO this is just a rearrangement of the data, is it really useful, or is there another way?
    comps = defaultdict(dict)
    for res, group in grp_vars.items():
      for variable in group:
        _, comp, tracker, res = variable.split('__')
        if res not in comps[comp]:
          comps[comp][res] = [tracker]
        else:
          comps[comp][res].append(tracker)
    n_comps = len(comps)
    n_colors = len(colormap)
    n_hatches = len(hatchmap)
    n_uniques = n_colors * n_hatches
    if n_comps > n_uniques:
      self.raiseAWarning(f'A total of {n_comps} exist to plot, but only {n_uniques} unique identifying ' +
                         'colors and patterns are available! This may lead to dispatch plot confusion.')
    print('Assigning colors ...')
    comp_colors = {}
    # kept for easy debugging
    #print('DEBUGG | c | ci | hi | comp | hatch | color_r | color_g | color_b')
    for c, (comp, ress) in enumerate(comps.items()):
      hatch_index, color_index = divmod(c, n_colors)
      color = colormap[color_index]
      hatch = hatchmap[hatch_index]
      # kept for easy debugging
      #print(f'DEBUGG | {c:2d} | {color_index:1d} | {hatch_index:1d} | {comp:20s} | '+
      #      f'{hatch if hatch is not None else "None":4s} | '+
      #      f'{color[0]*255:1.8f} | {color[1]*255:1.8f} | {color[2]*255:1.8f}')
      comp_colors[comp] = {'color': color, 'hatch': hatch}
    return comp_colors

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

    # Dimension variables to plot
    sample_ids = df[self._source.sampleTag].unique()
    cluster_ids = df['_ROM_Cluster'].unique()  # TODO: find way to not hardcode name
    macro_steps = df[self._macroName].unique()

    # Assign colors
    cdict = self.color_style(grouped_vars)

    resources = set([x for x in grouped_vars])
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

      # If only 3 resources, make one figure; otherwise, 2 resources per figure
      if len(resources) <= 3:
        fig, res_axs = plt.subplots(len(resources), 1, sharex=True, squeeze=False)
        res_figs = [fig]
      else:
        res_figs = []
        res_axs = []
        for _ in range(int(np.ceil(len(resources) / 2))):
          fig, axs = plt.subplots(2, 1, sharex=True, squeeze=False)
          res_figs.append(fig)
          res_axs.extend(axs)

      # Output optimized component dispatch for current dimension.
      res_axs = np.asarray(res_axs)
      self.plot_dispatch(res_figs, res_axs, dat, grouped_vars, sample_id, macro_step, cluster_id, cdict)

      # Output synthetic time series signal for current dimension.
      sig_figs, sig_axs = plt.subplots(len(self._addSignals), 1, sharex=True, squeeze=False)
      self.plot_signal(sig_figs, sig_axs, dat, sample_id, macro_step, cluster_id)

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

  def plot_component(self, fig, axes, df, grp_vars, comp_idx, sid, mstep, cid) -> None:
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
      @ Out, None
    """
    for (key, group), ax in zip(grp_vars.items(), axes.flat):
      lines = []
      for var in group:
        _, comp_name, tracker, _ = var.split('__')
        comp_label = comp_name.replace('_', ' ').title()
        cidx = comp_idx[comp_name]

        # NOTE custom behavior based on production/storage labels
        plot_ax = ax
        var_label = f'{comp_label}, {tracker.title()}'
        ls = '-'
        mk = '1'
        if tracker == 'production':
          var_label = comp_label
        elif tracker == 'level':
          plot_ax = ax.twinx()
          ls = ':'
          mk = '.'
        elif tracker == 'charge':
          mk = '^'
        elif tracker == 'discharge':
          mk = 'v'

        # Plot the micro-step variable on the x-axis (i.e Time)
        ln = plot_ax.plot(df[self._microName], df[var], marker=mk, linestyle=ls, label=var_label, color=f"C{cidx}")
        lines.extend(ln)
        ax.set_title(key.title())
        ax.set_xlabel(self._microName)
        ax.legend(lines, [l.get_label() for l in lines], loc='center left', bbox_to_anchor=(1.03, 0.5))

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
      self.plot_component(fig0, axs0, dat, grouped_vars, comp_idx, sample_id, macro_step, cluster_id)

      # Output synthetic time series signal for current dimension.
      fig1, axs1 = plt.subplots(len(self._addSignals), 1, sharex=True, squeeze=False)
      self.plot_signal(fig1, axs1, dat, sample_id, macro_step, cluster_id)

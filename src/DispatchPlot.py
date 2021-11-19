# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Author: dylanjm
  Date: 2021-05-18
"""
import os
import sys
import itertools as it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PluginBaseClasses.OutStreamPlotPlugin import PlotPlugin, InputTypes, InputData

# Matplotlib Global Settings
plt.rc("figure", figsize=(12, 8), titleweight='bold')
plt.rc(
  "axes",
  titleweight="bold",
  labelsize=12,
  axisbelow=True,
  grid=True
)
plt.rc("savefig", bbox="tight")
plt.rc("legend", fontsize=12)
plt.rc(["xtick", "ytick"], labelsize=10)


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
  def _group_by(iterable):
    """
      Returns dictionary containing grouped dispatch variables.
      @ In, iterable, list, a list of column names to group-by.
      @ Out, gr, dict, a dictionary containing a mapping of grouped variable names.
    """
    gr = {}
    for var in iterable:
      key = var.split('__')[-1]
      if key in gr.keys():
        gr[key].append(var)
      else:
        gr[key] = [var]
    return gr

  @staticmethod
  def _group_by_component(iterable):
    """
      Returns dictionary containing grouped dispatch variables.
      @ In, iterable, list, a list of column names to group-by.
      @ Out, gr, dict, a dictionary containing a mapping of grouped variable names.
    """
    gr = {}
    for var in iterable:
      key = var.split('__')[1]
      if key in gr.keys():
        gr[key].append(var)
      else:
        gr[key] = [var]
    return gr


  def run(self):
    """
      Generate the plot
      @ In, None
      @ Out, None
    """
    ds = self._source.asDataset()
    if ds is None:
      self.raiseAWarning(f'No data in "{self._source.name}" data object; nothing to plot!')
      return
    df = ds.to_dataframe().reset_index()
    dispatch_vars = list(filter(lambda x: "Dispatch__" in x, df.columns))
    grouped_vars = self._group_by(dispatch_vars)
    grouped_comp = self._group_by_component(dispatch_vars)
    comp_idx = dict((comp, i) for i,comp in enumerate(grouped_comp.keys()))

    # Dimension variables to plot with
    sample_ids = df[self._source.sampleTag].unique()
    cluster_ids = df['_ROM_Cluster'].unique()  # TODO: find way to not hardcode name
    macro_steps = df[self._macroName].unique()


    for sample_id in sample_ids:
      for macro_step in macro_steps:
        for cluster_id in cluster_ids:
          total_axs = len(grouped_vars) # + len(self._addSignals)
          fig = plt.figure()
          axs = fig.subplots(total_axs, 1, sharex=True)
          # Filter data to plot correct values for current dimension
          dat = df[
            (df[self._source.sampleTag] == sample_id) &
            (df[self._macroName] == macro_step) &
            (df['_ROM_Cluster'] == cluster_id)
          ]
          micro = dat[self._microName]
          # add dispatch lines
          for i, (key, group) in enumerate(grouped_vars.items()):
            ax = axs[i] #fig.add_subplot(total_axs, 1, i+1, sharex=True)
            ax_twin = ax.twinx()
            lines = []
            for var in group:
              _, comp_name, tracker, resource = var.split('__')
              cidx = comp_idx[comp_name]
              comp_label = comp_name.replace('_', ' ').title()

              # NOTE custom behavior based on production/storage labels
              if tracker == 'production':
                var_label = comp_label
              else:
                var_label = f'{comp_label}, {tracker.title()}'

              if tracker == 'level':
                plot_ax = ax_twin
                ls = ':'
              else:
                plot_ax = ax
                ls = '-'

              # Plot the micro-step variable on the x-axis (i.e Time)
              ln = plot_ax.plot(micro, dat[var],  marker='.', linestyle=ls, label=var_label, color=f"C{cidx}")
              lines.extend(ln)

            ax.set_title(key.title())
            ax.set_xlabel(self._microName.title())
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1.1, 0.5))

          file_name = f"dispatch_id{sample_id}_y{macro_step}_c{cluster_id}.png"
          fig.tight_layout()
          fig.savefig(file_name)
          self.raiseAMessage(f'Saved figure to "{file_name}"')
          plt.clf()

          # add signal lines (e.g. synthetic history values)
          total_axs = len(self._addSignals)
          fig02 = plt.figure()
          axs = fig02.subplots(total_axs, 1, sharex=True)
          for s, name in enumerate(self._addSignals):
            var = dat.get(name, None)
            if var is None:
              msg = f'Requested signal variable "{name}" but variable not in data!'
              self.raiseAnError(msg)
            ax = axs[s]
            ax.plot(micro, var, marker='.', linestyle='-', label=name)
            ax.set_title(name.title())
            ax.set_xlabel(self._microName.title())
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

          signal_file_name = f"dispatch_id{sample_id}_y{macro_step}_c{cluster_id}_SIGNAL.png"
          fig02.tight_layout()
          fig02.savefig(signal_file_name)
          self.raiseAMessage(f'Saved figure to "{signal_file_name}"')
          plt.clf()

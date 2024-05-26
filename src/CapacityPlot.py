# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on September 29, 2023

@author: yangx
"""
import itertools as it
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import random
import math

try:
  from ravenframework.PluginBaseClasses.OutStreamPlotPlugin import PlotPlugin, InputTypes, InputData
except ModuleNotFoundError:
  import sys
  from . import _utils
  sys.path.append(_utils.get_raven_loc())
  from ravenframework.PluginBaseClasses.OutStreamPlotPlugin import PlotPlugin, InputTypes, InputData

class CapacityPlot(PlotPlugin):
  """
    Plots the path that variables took during an optimization, including accepted and rejected runs.
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""The name of the RAVEN DataObject from which the data should be taken for this plotter.
              This should be the SolutionExport for a MultiRun with an Optimizer."""))
    spec.addSub(InputData.parameterInputFactory('vars', contentType=InputTypes.StringListType,
        descr=r"""Names of the variables from the DataObject whose optimization paths should be plotted."""))
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'OptPath Plot'
    self.source = None      # reference to DataObject source
    self.sourceName = None  # name of DataObject source
    self.vars = None        # variables to plot
    self.markerMap = {'first': 'yo',
                      'accepted': 'go',
                      'rejected': 'rx',
                      'rerun': 'c.',
                      'final': 'mo'}
    self.markers = defaultdict(lambda: 'k.', self.markerMap)

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value
    self.vars = spec.findFirst('vars').value
    # checker; this should be superceded by "required" in input params
    if self.sourceName is None:
      self.raiseAnError(IOError, "Missing <source> node!")
    if self.vars is None:
      self.raiseAnError(IOError, "Missing <vars> node!")

  def initialize(self, stepEntities):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system.
      @ In, stepEntities, dict, contains all the Objects are going to be used in the
                                current step. The sources are searched into this.
      @ Out, None
    """
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" was found in the Step for SamplePlot "{self.name}"!')
    self.source = src
    # sanity check
    dataVars = self.source.getVars()
    missing = [var for var in (self.vars+['accepted']) if var not in dataVars]
    if missing:
      msg = f'Source DataObject "{self.source.name}" is missing the following variables ' +\
            f'expected by OptPath plotter "{self.name}": '
      msg += ', '.join(f'"{m}"' for m in missing)
      self.raiseAnError(IOError, msg)

  def last_accepted_index(self, seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs[-1]

  def run(self):
    """
      Main run method.
      @ In, None
      @ Out, None
    """
    fig, ax = plt.subplots()
    # Get accepted results for all iterations
    accepted = [] # Final accepted run
    zero_capacity = [] # Zero capacity components in the final accepted run
    labels = []
    zero_labels = ''
    negative_exist = False
    for r in range(len(self.source)):
      rlz = self.source.realization(index=r, asDataSet=True, unpackXArray=False)
      accepted.append(rlz['accepted'])
    # Check if there's a final results. If not, get the last accepted run.
    # Find the index of plotting results
    i = None
    plot_data = {}
    if 'final' in accepted:
      i = accepted.index('final')
    elif 'accepted' in accepted:
      i = self.last_accepted_index(self, accepted,'accepted')
    if i is not None:
      data = self.source.realization(index=r, asDataSet=True, unpackXArray=False)
      for var in self.vars:
        if 'mean' in var:
          self.raiseAMessage(data[var])
          millnames = ['',' K',' MM',' B',' T'] # thousand, million, billion, trillion
          n = float(abs(round(data[var])))
          millidx = max(0,min(len(millnames)-1, int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
          currency_string = str('{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx]))
          mean_txt = 'Mean: ' + currency_string
          if data[var] > 0: # if mean_NPV is positive, color it red; otherwise, color it green
            c = 'green'
          else:
            c = 'red'
        else:
            if round(data[var],2) == 0: #components optimized to zero capacity
               zero_capacity.append(var)
            else:
               plot_data[var] = abs(data[var])
    # Generate visualization output if the total optimized capacity is not zero
    total_capacity = sum(abs(val) for val in plot_data.values())
    labels_lt1 = [] # labels for components capacity percentage < 1%
    index_lt1 = [] # index for components capacity percentage < 1% for legend
    i = 0 # index
    if total_capacity > 0:
        if len(plot_data.keys()) > 0:
            for var in plot_data.keys():
              val = data[var]
              if val < 0:
                negative_exist = True
              var = var.capitalize()
              var = var.replace('_', ' ')
              var = var.replace('capacity', '')
              var = var + '\n' + str(round(val,2))
              if abs(val / total_capacity * 100) >= 1:
                labels.append(var)
                #labels_lt1.append('')
              else: # components capacity less than 1% of the total
                labels_lt1.append(var)
                labels.append('')
                index_lt1.append(i)
              i = i + 1
        if len(zero_capacity) > 0:
            for zero_var in zero_capacity:
              zero_var = zero_var.capitalize()
              zero_var = zero_var.replace('_', ' ')
              zero_var = zero_var.replace('capacity', '')
              zero_var = zero_var + '\n'
              zero_labels += zero_var
            zero_labels = 'Component optimized to zero capacity:\n' + zero_labels
        #Text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        if len(plot_data.values()) > 0:
            cmap = plt.get_cmap('Pastel1')(np.arange(len(labels), dtype = int))
            # donut chart and mean NPV value in the center
            ax.pie(plot_data.values(), textprops={'fontsize': 13}, pctdistance=13,labels = labels, labeldistance=1.05, colors = cmap)
            ax.text(0, 0, mean_txt, fontsize = 15, va = 'center', ha = 'center', color = c)  # mean NPV
            if negative_exist == True:
                ax.text(0, -1.3, 'Negative values represent components that only consume\n resources and are still nominally part of the system size.', fontsize = 13, va = 'center', ha = 'center', color = 'grey')
            if len(labels_lt1) > 0:
                # Get components less than 1% legend color
                custom_legend = []
                for i in index_lt1:
                    custom_legend.append(Line2D([0], [0], color = cmap[i], lw = 10))
                plt.legend(custom_legend, labels_lt1, fontsize = 13, title_fontsize = 13, loc = 'center left', bbox_to_anchor = (1.2, 0.5), title = 'Components < 1% of Total System Capacity')
            if len(zero_capacity) > 0:
                yoffset = -1 * (1.6 + 0.1 * len(zero_capacity)) # adjust location for the text box of zero capacity components
                ax.text(0, yoffset, zero_labels, fontsize=13,va = 'bottom', ha = 'center', bbox=props)
            plt.title('Optimized Capacity', fontsize = 18)
            centre_circle = plt.Circle((0, 0), 0.60, fc='white')
            fig.gca().add_artist(centre_circle)
            plt.savefig(f'{self.name}.png')
        else:
            self.raiseAWarning(f'No optimized components; nothing to plot!')
    else:
        self.raiseAWarning(f'The total optimized capacity is zero; nothing to plot!')

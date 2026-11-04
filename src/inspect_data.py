# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Reports the time structure of a time series data source (a serialized RAVEN synthetic history ROM
  or a CSV) as HERON sees it, so the <Case> time discretization and <DataGenerators> can be set to match.
"""
import os
import pickle

import numpy as np

from HERON.src import _utils as hutils

def inspect_data(fpath, time_var=None, year_var=None):
  """
    Inspects a data source, choosing how by its file type
    @ In, fpath, str, path to a serialized ROM (.pk) or CSV file
    @ In, time_var, str, optional, CSV only: name of the time column
    @ In, year_var, str, optional, CSV only: name of the year column
    @ Out, info, dict, description of the data source (see inspect_rom and inspect_csv)
  """
  fpath = os.path.abspath(os.path.expanduser(fpath))
  if not os.path.isfile(fpath):
    raise FileNotFoundError(f'Data file not found: "{fpath}"')
  if fpath.lower().endswith('.csv'):
    return inspect_csv(fpath, time_var=time_var, year_var=year_var)
  return inspect_rom(fpath)

def _time_axis(values):
  """
    Summarizes the values of a time variable
    @ In, values, np.array, time values for one dispatch cycle
    @ Out, axis, dict, number of steps, first and last values, and step size (None if not uniform)
  """
  values = np.asarray(values, dtype=float).ravel()
  steps = np.diff(values)
  uniform = steps.size > 0 and np.allclose(steps, steps[0])
  return {
    'num_steps': int(values.size),
    'first': float(values[0]),
    'last': float(values[-1]),
    'interval': float(steps[0]) if uniform else None,
  }

def inspect_rom(fpath):
  """
    Inspects a serialized RAVEN ROM. Synthetic history ROMs are sampled once per cluster evaluation mode.
    @ In, fpath, str, absolute path to the serialized ROM
    @ Out, info, dict, description with keys
      'kind' ('synthetic_history' or 'rom'), 'path', 'features', 'targets', and for synthetic histories
      'signals', 'time_variable', 'year_variable', 'years', 'multiyear', 'clustered', and 'modes'
      ({eval mode: {'time': time axis, 'clusters': number of clusters}})
  """
  hutils._ensure_raven_importable()
  with open(fpath, 'rb') as f:
    rom = pickle.load(f)
  params = rom.getInitParams()
  features = list(params.get('Features', []))
  targets = list(params.get('Target', []))
  info = {'kind': 'rom', 'path': fpath, 'features': features, 'targets': targets}
  # synthetic history ROMs take only the "scaling" feature; anything else is a regular ROM (<ROM> generator)
  if features != ['scaling']:
    return info
  info['kind'] = 'synthetic_history'
  structure = hutils.get_synthhist_structure(fpath)
  info['clustered'] = any(structure['clusters'].values())
  # same test Placeholders.ARMA.checkValid uses to decide between truncating and <Multicycle>
  info['multiyear'] = 'macro' in structure
  # 'clustered' is the <ARMA> evalMode default; 'full' only differs for a clustered ROM
  modes = ['clustered', 'full'] if info['clustered'] else ['clustered']
  info['modes'] = {}
  for mode in modes:
    real = hutils.sample_synthhist(fpath, eval_mode=mode)
    index_map = real['_indexMap'][0]
    signals = list(index_map)
    # every signal is indexed the same way: [year,] [_ROM_Cluster,] time
    indices = index_map[signals[0]]
    time_var = indices[-1]
    year_var = indices[0] if len(indices) > 1 and indices[0] != '_ROM_Cluster' else None
    info['signals'] = signals
    info['time_variable'] = time_var
    info['year_variable'] = year_var
    info['years'] = [int(y) for y in real[year_var]] if year_var else []
    info['modes'][mode] = {
      'time': _time_axis(real[time_var]),
      'clusters': int(np.asarray(real['_ROM_Cluster']).size) if '_ROM_Cluster' in real else 1,
    }
  return info

def inspect_csv(fpath, time_var=None, year_var=None):
  """
    Inspects a CSV static history
    @ In, fpath, str, absolute path to the CSV
    @ In, time_var, str, optional, name of the time column; found by name ("time", "hour") if not given
    @ In, year_var, str, optional, name of the year column; found by name ("year") if not given
    @ Out, info, dict, description with keys 'kind' ('csv'), 'path', 'columns', 'signals',
      'time_variable', 'year_variable', 'years', 'multiyear', 'num_samples', and 'time' (time axis)
  """
  import pandas as pd
  data = pd.read_csv(fpath, encoding='utf-8-sig')
  columns = [c.strip() for c in data.columns]
  data.columns = columns
  def find(given, names, what):
    """
      Finds a column, either the one given or the first whose lowercase name is in names
      @ In, given, str or None, column requested by the user
      @ In, names, list(str), lowercase names to look for
      @ In, what, str, description of the column for error messages
      @ Out, found, str or None, column name
    """
    if given is not None:
      if given not in columns:
        raise KeyError(f'{what} column "{given}" not found in "{fpath}". Columns: {columns}')
      return given
    return next((c for c in columns if c.lower() in names), None)
  time_var = find(time_var, ['time', 'hour'], 'Time')
  if time_var is None:
    raise KeyError(f'Could not identify the time column of "{fpath}"; name it with --time-variable. Columns: {columns}')
  year_var = find(year_var, ['year'], 'Year')
  if 'RAVEN_sample_ID' not in columns:
    raise KeyError(f'"{fpath}" has no RAVEN_sample_ID column, which HERON requires in CSV sources.')
  groups = ['RAVEN_sample_ID'] + ([year_var] if year_var else [])
  lengths = data.groupby(groups)[time_var].size().unique()
  if lengths.size != 1:
    raise ValueError(f'Each sample and year of "{fpath}" must have the same number of time steps; found {sorted(lengths)}.')
  first = data.groupby(groups).get_group(tuple(data[groups].iloc[0]))
  reserved = {'RAVEN_sample_ID', '_ROM_Cluster', 'scaling', 'ProbabilityWeight', 'prefix', 'PointProbability', time_var, year_var}
  return {
    'kind': 'csv',
    'path': fpath,
    'columns': columns,
    'signals': [c for c in columns if c not in reserved],
    'time_variable': time_var,
    'year_variable': year_var,
    'years': sorted(int(y) for y in data[year_var].unique()) if year_var else [],
    # Placeholders.CSV.checkValid treats any CSV with a year column as multi-year
    'multiyear': year_var is not None,
    'num_samples': int(data['RAVEN_sample_ID'].unique().size),
    'time': _time_axis(first[time_var].values),
  }

def _fmt(value):
  """
    Formats a number for XML, dropping a trailing ".0"
    @ In, value, float, number
    @ Out, fmt, str, formatted number
  """
  return f'{value:g}'

def _time_discretization_xml(time_var, year_var, axis, indent='    '):
  """
    Writes the <time_discretization> that matches a time axis
    @ In, time_var, str, time variable name
    @ In, year_var, str or None, year variable name
    @ In, axis, dict, time axis from _time_axis
    @ In, indent, str, optional, prefix for each line
    @ Out, xml, str, the XML node
  """
  lines = ['<time_discretization>', f'  <time_variable>{time_var}</time_variable>']
  if year_var:
    lines.append(f'  <year_variable>{year_var}</year_variable>')
  lines += [
    f'  <start_time>{_fmt(axis["first"])}</start_time>',
    f'  <end_time>{_fmt(axis["last"])}</end_time>',
    f'  <num_steps>{axis["num_steps"]}</num_steps>',
    '</time_discretization>',
  ]
  return '\n'.join(indent + line for line in lines)

def _time_axis_text(axis):
  """
    Describes a time axis in one line
    @ In, axis, dict, time axis from _time_axis
    @ Out, text, str, description
  """
  step = _fmt(axis['interval']) if axis['interval'] is not None else 'not uniform'
  return f'{axis["num_steps"]} steps from {_fmt(axis["first"])} to {_fmt(axis["last"])} (interval {step})'

def _years_text(info):
  """
    Describes the years a data source covers and what that allows for <ProjectTime>
    @ In, info, dict, description from inspect_rom or inspect_csv
    @ Out, text, list(str), report lines
  """
  years = info['years']
  if info['multiyear']:
    return [f'Years ({info["year_variable"]}): {len(years)}, {years[0]} to {years[-1]}',
            f'  -> each project year uses its own data year; <ProjectTime> should not exceed {len(years)}.']
  return ['Years: one cycle of data',
          '  -> the same cycle is repeated for every project year; any <ProjectTime> works.']

def format_report(info):
  """
    Writes a human-readable report of an inspected data source
    @ In, info, dict, description from inspect_data
    @ Out, report, str, the report
  """
  lines = [f'Data source: {info["path"]}']
  if info['kind'] == 'rom':
    lines += [
      'Type: RAVEN ROM (not a synthetic history), for a <ROM> data generator',
      f'Features (inputs): {", ".join(info["features"])}',
      f'Targets (outputs): {", ".join(info["targets"])}',
    ]
    return '\n'.join(lines)
  if info['kind'] == 'csv':
    lines += [
      'Type: CSV static history, for a <CSV> data generator',
      f'Signals: {", ".join(info["signals"])}',
      f'Time variable: {info["time_variable"]}',
      f'Samples (RAVEN_sample_ID): {info["num_samples"]}',
    ]
    lines += _years_text(info)
    lines += [
      f'Time axis: {_time_axis_text(info["time"])}',
      '',
      'Matching <Case> settings:',
      _time_discretization_xml(info['time_variable'], info['year_variable'], info['time']),
    ]
    return '\n'.join(lines)
  # synthetic history
  lines += [
    'Type: synthetic history ROM, for an <ARMA> data generator',
    f'Signals: {", ".join(info["signals"])}',
    f'Time variable: {info["time_variable"]}',
    f'Clustered: {"yes" if info["clustered"] else "no"}',
  ]
  lines += _years_text(info)
  for mode, details in info['modes'].items():
    note = ' (default)' if mode == 'clustered' else ''
    lines += [
      '',
      f'evalMode="{mode}"{note}:',
      f'  Time axis: {_time_axis_text(details["time"])}',
    ]
    if info['clustered']:
      lines.append(f'  Clusters dispatched per year: {details["clusters"]}')
    lines += [
      '  Matching <Case> settings:',
      _time_discretization_xml(info['time_variable'], info['year_variable'], details['time']),
    ]
  return '\n'.join(lines)

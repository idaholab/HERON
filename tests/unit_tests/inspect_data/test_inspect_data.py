# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Unit tests for inspecting time series data sources (heron --inspect-data)
"""

import sys
import os
import unittest

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.src import inspect_data
sys.path.pop()

DATA = os.path.join(HERON_LOC, 'HERON', 'data')


class TestInspectData(unittest.TestCase):
  """ Checks the structure reported for the ROMs and CSVs shipped in HERON/data """

  def test_clustered_multiyear_arma(self):
    """
      A clustered, interpolated synthetic history reports both evaluation modes
      @ In, None
      @ Out, None
    """
    info = inspect_data.inspect_data(os.path.join(DATA, 'TSA', 'NYISO', 'nyiso_arma_2yr.pk'))
    self.assertEqual(info['kind'], 'synthetic_history')
    self.assertEqual(info['signals'], ['TOTALLOAD', 'WIND', 'SOLAR'])
    self.assertEqual(info['time_variable'], 'HOUR')
    self.assertEqual(info['year_variable'], 'YEAR')
    self.assertEqual(info['years'], [2020, 2021, 2022])
    self.assertTrue(info['multiyear'])
    self.assertTrue(info['clustered'])
    clustered = info['modes']['clustered']
    self.assertEqual(clustered['clusters'], 2)
    self.assertEqual(clustered['time'], {'num_steps': 24, 'first': 0.0, 'last': 23.0, 'interval': 1.0})
    full = info['modes']['full']
    self.assertEqual(full['time'], {'num_steps': 8760, 'first': 1.0, 'last': 8760.0, 'interval': 1.0})

  def test_fractional_time_arma(self):
    """
      A non-integer time step is reported exactly
      @ In, None
      @ Out, None
    """
    info = inspect_data.inspect_data(os.path.join(DATA, 'TSA', 'Sine_Hour', 'arma.pk'))
    self.assertEqual(info['time_variable'], 'Hour')
    axis = info['modes']['clustered']['time']
    self.assertEqual(axis['num_steps'], 21)
    self.assertAlmostEqual(axis['interval'], 0.1)

  def test_regular_rom(self):
    """
      A ROM that is not a synthetic history is identified, not sampled
      @ In, None
      @ Out, None
    """
    info = inspect_data.inspect_data(os.path.join(DATA, 'ROM', 'linear_rom.pk'))
    self.assertEqual(info['kind'], 'rom')
    self.assertEqual(info['features'], ['a', 'b', 'c', 'd'])
    self.assertEqual(info['targets'], ['price'])

  def test_csv(self):
    """
      A multi-year CSV reports its signals, years, and time axis
      @ In, None
      @ Out, None
    """
    info = inspect_data.inspect_data(os.path.join(DATA, 'CSV', '2year_21step.csv'))
    self.assertEqual(info['kind'], 'csv')
    self.assertEqual(info['signals'], ['spike', 'linear'])
    self.assertEqual(info['time_variable'], 'Time')
    self.assertEqual(info['year_variable'], 'Year')
    self.assertEqual(info['years'], [10, 11])
    self.assertEqual(info['num_samples'], 1)
    self.assertEqual(info['time']['num_steps'], 21)
    self.assertAlmostEqual(info['time']['last'], 2.0)

  def test_report_time_discretization(self):
    """
      The report contains a <time_discretization> matching the data
      @ In, None
      @ Out, None
    """
    info = inspect_data.inspect_data(os.path.join(DATA, 'CSV', '2year_21step.csv'))
    report = inspect_data.format_report(info)
    self.assertIn('<time_variable>Time</time_variable>', report)
    self.assertIn('<end_time>2</end_time>', report)
    self.assertIn('<num_steps>21</num_steps>', report)

  def test_missing_file(self):
    """
      A missing file is a clear error
      @ In, None
      @ Out, None
    """
    with self.assertRaises(FileNotFoundError):
      inspect_data.inspect_data(os.path.join(DATA, 'does_not_exist.pk'))


if __name__ == '__main__':
  unittest.main()

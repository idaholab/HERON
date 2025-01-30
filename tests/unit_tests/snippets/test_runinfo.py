"""
Unit tests for the RunInfo snippets
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.templates.snippets import RunInfo
from HERON.templates.snippets.runinfo import get_default_parallel_settings, get_parallel_xml
from HERON.templates.snippets import IOStep
sys.path.pop()

import unittest
import xml.etree.ElementTree as ET


class TestRunInfo(unittest.TestCase):
  """ Tests for the RunInfo RavenSnippet class """

  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.run_info = RunInfo()

    xml = """
    <RunInfo>
      <JobName>test_job_name</JobName>
      <WorkingDir>./working/dir/name</WorkingDir>
      <Sequence>step1, step2</Sequence>
      <batchSize>2</batchSize>
    </RunInfo>
    """
    root = ET.fromstring(xml)
    run_info_xml = RunInfo.from_xml(root)
    self.run_info_xml = run_info_xml

  def test_from_xml(self):
    """
    Test instantiate from XML
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.run_info_xml.job_name, "test_job_name")
    self.assertEqual(self.run_info_xml.working_dir, "./working/dir/name")
    self.assertEqual(self.run_info_xml.batch_size, 2)
    self.assertListEqual(self.run_info_xml.sequence, ["step1", "step2"])

  def test_add_step_to_sequence(self):
    """
    Test append step to sequence
    @ In, None
    @ Out, None
    """
    self.run_info_xml.sequence.append("new_step")
    self.assertListEqual(self.run_info_xml.sequence, ["step1", "step2", "new_step"])

  def test_snippet_class(self):
    """
    Test snippet_class value
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.run_info.snippet_class)

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.run_info.tag, "RunInfo")

  def test_job_name(self):
    """
    Test job_name property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.run_info.find("JobName"))
    self.assertIsNone(self.run_info.job_name)
    job_name = "some_job_name"
    self.run_info.job_name = job_name
    self.assertEqual(self.run_info.job_name, job_name)
    self.assertEqual(self.run_info.find("JobName").text, job_name)

  def test_working_dir(self):
    """
    Test working_dir property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.run_info.find("WorkingDir"))
    self.assertIsNone(self.run_info.working_dir)
    working_dir = "path/to/dir"
    self.run_info.working_dir = working_dir
    self.assertEqual(self.run_info.working_dir, working_dir)
    self.assertEqual(self.run_info.find("WorkingDir").text, working_dir)

  def test_batch_size(self):
    """
    Test batch_size property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.run_info.find("batchSize"))
    self.assertIsNone(self.run_info.batch_size)
    batch_size = 10
    self.run_info.batch_size = batch_size
    self.assertEqual(self.run_info.batch_size, batch_size)
    self.assertEqual(self.run_info.find("batchSize").text, batch_size)

  def test_internal_parallel(self):
    """
    Test internal_parallel property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.run_info.find("internalParallel"))
    self.assertFalse(self.run_info.use_internal_parallel)

    # Set to True
    internal_parallel = True
    self.run_info.use_internal_parallel = internal_parallel
    self.assertEqual(self.run_info.use_internal_parallel, internal_parallel)
    self.assertEqual(self.run_info.find("internalParallel").text, internal_parallel)

    # Set to False
    internal_parallel = False
    self.run_info.use_internal_parallel = internal_parallel
    self.assertEqual(self.run_info.use_internal_parallel, internal_parallel)
    self.assertIsNone(self.run_info.find("internalParallel"))

  def test_num_mpi(self):
    """
    Test num_mpi property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.run_info.find("NumMPI"))
    self.assertIsNone(self.run_info.num_mpi)
    num_mpi = 10
    self.run_info.num_mpi = num_mpi
    self.assertEqual(self.run_info.num_mpi, num_mpi)
    self.assertEqual(self.run_info.find("NumMPI").text, num_mpi)

  def test_sequence(self):
    """
    Test sequence property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.run_info.find("Sequence"))
    self.assertListEqual(self.run_info.sequence, [])
    self.run_info.sequence.append("step1")
    self.run_info.sequence.extend(["step2", "step3"])
    self.run_info.sequence.insert(0, "step0")
    self.assertListEqual(self.run_info.sequence, ["step0", "step1", "step2", "step3"])
    self.run_info.sequence.clear()
    self.assertListEqual(self.run_info.sequence, [])

  def test_add_step_obj_to_sequence(self):
    """
    Test add Step object to sequence
    @ In, None
    @ Out, None
    """
    iostep = IOStep(name="new_step")
    self.run_info_xml.sequence.append(iostep)
    self.assertListEqual(self.run_info_xml.sequence, ["step1", "step2", "new_step"])
    self.assertListEqual(self.run_info_xml.find("Sequence").text, ["step1", "step2", "new_step"])

  def test_set_parallel_run_settings(self):
    """
    Test set_parallel_run_settings method
    @ In, None
    @ Out, None
    """
    parallel_info = {
      "memory": "4g",
      "expectedTime": "72:0:0",
      "clusterParameters": "-P nst"
    }
    self.run_info.set_parallel_run_settings(parallel_info)

    for k, v in parallel_info.items():
      node = self.run_info.find(k)
      self.assertIsNotNone(node)
      self.assertEqual(node.text, v)

  def test_apply_parallel_xml(self):
    """
    Test apply_parallel_xml method
    @ In, None
    @ Out, None
    """
    # Get the parallel XML settings to apply. We'll use the bitterroot settings to test.
    hostname = "bitterroot1.ib"
    xml = get_parallel_xml(hostname)
    self.run_info._apply_parallel_xml(xml)

    # Check if nodes have been inserted properly
    for gold in xml.find("useParallel"):
      test = self.run_info.find(gold.tag)
      self.assertEqual(ET.tostring(test), ET.tostring(gold))

    for gold in xml.find("outer"):
      test = self.run_info.find(gold.tag)
      self.assertEqual(ET.tostring(test), ET.tostring(gold))


class TestUtilities(unittest.TestCase):
  """ Tests for utility functions used by the RunInfo class """

  def test_get_default_parallel_settings(self):
    """
    Test get_default_parallel_settings function
    @ In, None
    @ Out, None
    """
    defaults = get_default_parallel_settings()
    gold = ET.fromstring("<parallel><useParallel><mode>mpi<runQSUB/></mode></useParallel></parallel>")
    self.assertEqual(ET.tostring(defaults), ET.tostring(gold))

  def test_get_parallel_xml(self):
    """
    Test get_parallel_xml function
    @ In, None
    @ Out, None
    """
    for cluster_name in ["sawtooth", "bitterroot"]:
      gold_xml_path = os.path.join(HERON_LOC, "HERON", "templates", "parallel", f"{cluster_name}.xml")
      gold = ET.tostring(ET.parse(gold_xml_path).getroot())
      for num in range(1, 3):
        hostname = f"{cluster_name}{num}.ib"
        xml = get_parallel_xml(hostname)
        self.assertEqual(ET.tostring(xml), gold)

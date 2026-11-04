# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Copies the agent skills packaged with HERON into a skills directory
  (e.g. ~/.claude/skills) so AI coding agents can use them.
"""
import os
import shutil

SKILLS_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'skills'))

def list_skills():
  """
    Finds the skills packaged with HERON. A skill is a directory containing a SKILL.md file.
    @ In, None
    @ Out, skills, list(str), names of available skills
  """
  if not os.path.isdir(SKILLS_SOURCE):
    return []
  return sorted(name for name in os.listdir(SKILLS_SOURCE)
                if os.path.isfile(os.path.join(SKILLS_SOURCE, name, 'SKILL.md')))

def install_skills(destination):
  """
    Copies each packaged skill into the destination directory. An existing skill
    directory with the same name is replaced, so re-running updates the skills.
    @ In, destination, str, directory to install the skills into
    @ Out, installed, list(str), paths of the installed skill directories
  """
  skills = list_skills()
  if not skills:
    raise FileNotFoundError(f'No HERON skills found in "{SKILLS_SOURCE}"')
  destination = os.path.abspath(os.path.expanduser(destination))
  os.makedirs(destination, exist_ok=True)
  installed = []
  for name in skills:
    target = os.path.join(destination, name)
    if os.path.isdir(target):
      shutil.rmtree(target)
    shutil.copytree(os.path.join(SKILLS_SOURCE, name), target)
    installed.append(target)
  return installed

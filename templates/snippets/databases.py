# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Database snippets

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
from ..xml_utils import find_node
from ..decorators import listproperty
from .base import RavenSnippet


class Database(RavenSnippet):
  """ Database snippet base class """
  snippet_class = "Databases"

  @property
  def read_mode(self) -> str | None:
    """
    Read mode getter
    @ In, None
    @ Out, read_mode, str | None, the database read mode
    """
    return self.get("readMode", None)

  @read_mode.setter
  def read_mode(self, value: str) -> None:
    """
    Read mode setter
    @ In, value, str, the database read mode
    @ Out, None
    """
    self.set("readMode", value)

  @property
  def directory(self) -> str | None:
    """
    Directory getter
    @ In, None
    @ Out, directory, str | None, the database directory
    """
    return self.get("directory", None)

  @directory.setter
  def directory(self, value: str) -> None:
    """
    Directory setter
    @ In, value, str, the database directory
    @ Out, None
    """
    self.set("directory", value)

  @property
  def filename(self) -> str | None:
    """
    File name getter
    @ In, None
    @ Out, filename, str | None, the database file name
    """
    return self.get("filename", None)

  @filename.setter
  def filename(self, value: str) -> None:
    """
    File name setter
    @ In, value, str, the database file name
    @ Out, None
    """
    self.set("filename", value)

  @listproperty
  def variables(self) -> list[str]:
    """
    Database variables getter
    @ In, None
    @ Out, variables, str | None, the database variables
    """
    node = self.find("variables")
    return getattr(node, "text", [])

  @variables.setter
  def variables(self, value: list[str]) -> None:
    """
    Database variables setter
    @ In, value, str, the database variables
    @ Out, None
    """
    find_node(self, "variables").text = value

  #####################
  # Getters & Setters #
  #####################
  def add_variable(self, *variables: str) -> None:
    """
    Add variables to the database
    @ In, *variables, str, variable names
    @ Out, None
    """
    self.variables.update(variables)


class NetCDF(Database):
  """ NetCDF database snippet class """
  tag = "NetCDF"


class HDF5(Database):
  """ HDF5 database snippet class """
  tag = "HDF5"

  @property
  def compression(self) -> str | None:
    """
    Compression getter
    @ In, None
    @ Out, compression, str | None, compression method
    """
    return self.get("compression", None)

  @compression.setter
  def compression(self, value: str) -> None:
    """
    Compression setter
    @ In, value, str, compression method
    @ Out, None
    """
    self.set("compression", value)

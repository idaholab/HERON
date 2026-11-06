# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Message handling interface for HERON objects, independent of RAVEN.

  A message backend decides whether and how messages are printed. Any object with the "message" and
  "error" methods of MessageBackend can be one, including RAVEN's MessageHandler, so RAVEN's handler
  can be used without HERON classes depending on it. HERON objects send messages through a Messenger,
  which provides the raiseA* methods and passes its owner to the backend as the caller.
"""
from typing import Protocol


class MessageBackend(Protocol):
  """
    Methods a message backend must provide. These match RAVEN's MessageHandler.
  """
  def message(self, caller, message, tag, verbosity, color=None, forcePrint=False):
    """
      Prints a message if the verbosity allows it
      @ In, caller, object, the object sending the message, used to label it
      @ In, message, str, message text
      @ In, tag, str, message label (e.g. 'Message', 'Warning', 'DEBUG')
      @ In, verbosity, str, priority of the message: 'silent', 'quiet', 'all', or 'debug'
      @ In, color, str, optional, color for the message
      @ In, forcePrint, bool, optional, if True print regardless of verbosity
      @ Out, None
    """

  def error(self, caller, etype, message, tag='ERROR', verbosity='silent', color=None):
    """
      Prints an error message, then raises the error
      @ In, caller, object, the object sending the message, used to label it
      @ In, etype, type, Exception class to raise
      @ In, message, str, error text
      @ In, tag, str, optional, message label
      @ In, verbosity, str, optional, priority of the message
      @ In, color, str, optional, color for the message
      @ Out, None
    """


def _join(args):
  """
    Joins message parts with spaces, as print() does
    @ In, args, tuple, message parts
    @ Out, text, str, joined message
  """
  return ' '.join(str(a) for a in args)


class Messenger:
  """
    The raiseA* message methods for one object (its owner), sent to a message backend.
    Owners create these on demand instead of storing one: a stored Messenger would make the owner refer
    to itself, and pickling such an object fails if its __getstate__ deep-copies its __dict__, as
    Placeholders.Function does.
  """
  def __init__(self, backend: MessageBackend, owner):
    """
      Constructor.
      @ In, backend, MessageBackend, formats and prints the messages
      @ In, owner, object, object named as the caller of each message
      @ Out, None
    """
    self._backend = backend
    self._owner = owner

  def raiseAnError(self, etype, *args, verbosity='silent', tag='ERROR', color=None):
    """
      Prints an error message and raises the error. By default shows at all verbosity levels.
      @ In, etype, type, Exception class to raise (e.g. IOError)
      @ In, args, tuple, message parts, joined with spaces as print() does
      @ In, verbosity, str, optional, priority of the message
      @ In, tag, str, optional, message label
      @ In, color, str, optional, color for the message
      @ Out, None
    """
    self._backend.error(self._owner, etype, _join(args), str(tag), verbosity, color)

  def raiseAWarning(self, *args, verbosity='quiet', tag='Warning', color=None):
    """
      Prints a warning. By default shows at verbosity 'quiet', 'all', and 'debug'.
      @ In, args, tuple, message parts, joined with spaces as print() does
      @ In, verbosity, str, optional, priority of the message
      @ In, tag, str, optional, message label
      @ In, color, str, optional, color for the message
      @ Out, None
    """
    self._backend.message(self._owner, _join(args), str(tag), verbosity, color)

  def raiseAMessage(self, *args, verbosity='all', tag='Message', color=None, forcePrint=False):
    """
      Prints a message. By default shows at verbosity 'all' and 'debug'.
      @ In, args, tuple, message parts, joined with spaces as print() does
      @ In, verbosity, str, optional, priority of the message
      @ In, tag, str, optional, message label
      @ In, color, str, optional, color for the message
      @ In, forcePrint, bool, optional, if True print regardless of verbosity
      @ Out, None
    """
    self._backend.message(self._owner, _join(args), str(tag), verbosity, color, forcePrint=forcePrint)

  def raiseADebug(self, *args, verbosity='debug', tag='DEBUG', color=None):
    """
      Prints a debug message. By default shows only at verbosity 'debug'.
      @ In, args, tuple, message parts, joined with spaces as print() does
      @ In, verbosity, str, optional, priority of the message
      @ In, tag, str, optional, message label
      @ In, color, str, optional, color for the message
      @ Out, None
    """
    self._backend.message(self._owner, _join(args), str(tag), verbosity, color)

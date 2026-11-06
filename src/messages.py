# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Message handling interface for HERON objects, independent of RAVEN.

  A message backend decides whether and how messages are printed. Any object with the "message" and
  "error" methods of MessageBackend can be one, including RAVEN's MessageHandler, so RAVEN's handler
  can be used without HERON classes depending on it. HERON objects send messages through a Messenger,
  which provides the raiseA* methods and passes its owner to the backend as the caller.

  HeronMessageHandler is HERON's own backend. Objects not given a backend use the default backend
  (see get_default_backend and set_default_backend).
"""
import sys
import time
import collections
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


# verbosity names and levels; a message prints if its level is at most the handler's level
VERBOSITY_LEVELS = {'silent': 0, # errors only
                    'quiet': 1,  # errors and warnings
                    'all': 2,    # errors, warnings, and messages
                    'debug': 3}  # everything, and errors show their traceback

_START_TIME = time.time() # message timestamps count seconds from here, as RAVEN's MessageHandler does

def _verbosity_level(verbosity):
  """
    Converts a verbosity name to its level
    @ In, verbosity, str, verbosity name, one of VERBOSITY_LEVELS (any case)
    @ Out, level, int, verbosity level
  """
  name = str(verbosity).strip().lower()
  if name not in VERBOSITY_LEVELS:
    # IOError, as RAVEN's MessageHandler raises
    raise IOError(f'Verbosity "{verbosity}" not recognized! Options are {list(VERBOSITY_LEVELS)}')
  return VERBOSITY_LEVELS[name]


class HeronMessageHandler:
  """
    HERON's message backend. Prints messages to standard output in the same format as RAVEN's
    MessageHandler, filtered by verbosity, and counts warnings so that a summary can be printed
    before an error is raised. Unlike RAVEN's MessageHandler, it does not color messages (the
    "color" arguments are accepted and ignored) and cannot suppress errors.
  """
  def __init__(self, verbosity='all', caller_length=25, tag_length=15, print_time=True):
    """
      Constructor. The defaults match those of RAVEN's MessageHandler.
      @ In, verbosity, str, optional, which messages print, one of VERBOSITY_LEVELS
      @ In, caller_length, int, optional, width of the column naming who sent the message
      @ In, tag_length, int, optional, width of the column for the message tag
      @ In, print_time, bool, optional, if True each message starts with the seconds since HERON started
      @ Out, None
    """
    self._level = _verbosity_level(verbosity)
    self._caller_length = caller_length
    self._tag_length = tag_length
    self._print_time = print_time
    self._warnings = collections.Counter() # warning text -> number of times raised

  def message(self, caller, message, tag, verbosity, color=None, forcePrint=False):
    """
      Prints a message if the verbosity allows it. Warnings (tag 'Warning') are counted even if not printed.
      @ In, caller, object, the object sending the message, named in the message
      @ In, message, str, message text
      @ In, tag, str, message label (e.g. 'Message', 'Warning', 'DEBUG')
      @ In, verbosity, str, priority of the message, one of VERBOSITY_LEVELS
      @ In, color, str, optional, ignored
      @ In, forcePrint, bool, optional, if True print regardless of verbosity
      @ Out, None
    """
    level = _verbosity_level(verbosity)
    if tag.strip().lower() == 'warning':
      self._warnings[message] += 1
    if forcePrint or level <= self._level:
      print(self._format(caller, tag, message), flush=True)

  def error(self, caller, etype, message, tag='ERROR', verbosity='silent', color=None):
    """
      Prints an error message and the warning summary, then raises the error. Below 'debug' verbosity,
      the traceback is hidden, so the user sees only the message.
      @ In, caller, object, the object sending the message, named in the message
      @ In, etype, type, Exception class to raise
      @ In, message, str, error text
      @ In, tag, str, optional, message label
      @ In, verbosity, str, optional, priority of the message, one of VERBOSITY_LEVELS
      @ In, color, str, optional, ignored
      @ Out, None
    """
    self.message(caller, message, tag, verbosity)
    self.print_warnings()
    if self._level < VERBOSITY_LEVELS['debug']:
      sys.tracebacklimit = 0
    raise etype(message)

  def print_warnings(self):
    """
      Prints a summary of the warnings counted so far, then clears them so they are not printed again.
      @ In, None
      @ Out, None
    """
    if not self._warnings:
      return
    total = sum(self._warnings.values())
    if self._level > VERBOSITY_LEVELS['silent']:
      print('-'*50)
      print(f'There were {total} warnings during the simulation run:')
      for text, count in sorted(self._warnings.items()):
        print(f'({count} {"time" if count == 1 else "times"}) {text}')
      print('-'*50)
    else:
      print(f'There were {total} warnings during the simulation run.')
    self._warnings.clear()

  def _format(self, caller, tag, text):
    """
      Formats a message line: "(time sec) caller : tag -> text", with the caller and tag in fixed-width columns
      @ In, caller, object, the object sending the message; its str() names it
      @ In, tag, str, message label
      @ In, text, str, message text
      @ Out, line, str, formatted message
    """
    name = str(caller).ljust(self._caller_length)[:self._caller_length]
    label = tag.ljust(self._tag_length)[:self._tag_length]
    line = f'{name}: {label} -> {text}'
    if self._print_time:
      line = f'({time.time() - _START_TIME:8.2f} sec) {line}'
    return line


_default_backend = None # backend for objects not given one; created when first needed

def get_default_backend():
  """
    Returns the message backend for objects that were not given one. Unless one was set with
    set_default_backend, this is a HeronMessageHandler with default settings.
    @ In, None
    @ Out, backend, MessageBackend, the default message backend
  """
  global _default_backend
  if _default_backend is None:
    _default_backend = HeronMessageHandler()
  return _default_backend

def set_default_backend(backend):
  """
    Sets the message backend for objects that were not given one
    @ In, backend, MessageBackend, the new default, or None to return to a HeronMessageHandler with default settings
    @ Out, None
  """
  global _default_backend
  _default_backend = backend

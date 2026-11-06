
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Base classes for other classes.
"""
from HERON.src.messages import Messenger, get_default_backend


class Base:
  """
    Base class for HERON objects. Sends messages through the message backend given as "messageHandler"
    (see HERON.src.messages), or through the default backend if none was given.
  """
  def __init__(self, **kwargs):
    """
      Constructor.
      @ In, kwargs, dict, passthrough arguments
      @ Out, None
    """
    # super().__init__() is not called, so that subclasses with more than one base class (e.g. Component)
    # call the other constructors themselves, as they did when Base derived from RAVEN's MessageUser.
    self.messageHandler = kwargs.get('messageHandler', None) # message backend, see HERON.src.messages

  @property
  def message_handler(self):
    """
      Message methods (raiseAnError, raiseAWarning, raiseAMessage, raiseADebug) for this object.
      @ In, None
      @ Out, message_handler, Messenger, sends this object's messages to its message backend
    """
    # the default is looked up each time, not stored, so it is not pickled with this object
    backend = self.messageHandler if self.messageHandler is not None else get_default_backend()
    return Messenger(backend, self)

  def raiseAnError(self, etype, *args, **kwargs):
    """
      Prints an error message and raises the error. Same as self.message_handler.raiseAnError.
      @ In, etype, type, Exception class to raise (e.g. IOError)
      @ In, args, tuple, message parts, joined with spaces as print() does
      @ In, kwargs, dict, optional settings (verbosity, tag, color), see Messenger.raiseAnError
      @ Out, None
    """
    self.message_handler.raiseAnError(etype, *args, **kwargs)

  def raiseAWarning(self, *args, **kwargs):
    """
      Prints a warning. Same as self.message_handler.raiseAWarning.
      @ In, args, tuple, message parts, joined with spaces as print() does
      @ In, kwargs, dict, optional settings (verbosity, tag, color), see Messenger.raiseAWarning
      @ Out, None
    """
    self.message_handler.raiseAWarning(*args, **kwargs)

  def raiseAMessage(self, *args, **kwargs):
    """
      Prints a message. Same as self.message_handler.raiseAMessage.
      @ In, args, tuple, message parts, joined with spaces as print() does
      @ In, kwargs, dict, optional settings (verbosity, tag, color, forcePrint), see Messenger.raiseAMessage
      @ Out, None
    """
    self.message_handler.raiseAMessage(*args, **kwargs)

  def raiseADebug(self, *args, **kwargs):
    """
      Prints a debug message. Same as self.message_handler.raiseADebug.
      @ In, args, tuple, message parts, joined with spaces as print() does
      @ In, kwargs, dict, optional settings (verbosity, tag, color), see Messenger.raiseADebug
      @ Out, None
    """
    self.message_handler.raiseADebug(*args, **kwargs)

  def __repr__(self):
    """
      String representation.
      @ In, None
      @ Out, repr, str rep
    """
    return f'<HERON {self.__class__.__name__}>'

  def set_message_handler(self, mh):
    """
      Sets message handling tool.
      @ In, mh, MessageHandler, message handler
      @ Out, None
    """
    self.messageHandler = mh

  def print_me(self, **kwargs):
    """
      Prints info about self.
      @ In, kwargs, dict, passthrough args
      @ Out, None
    """
    print(f'For {self} no printing has been implemented!')

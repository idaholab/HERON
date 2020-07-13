
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Base classes for other classes.
"""
import os
import sys

import _utils as hutils
raven_path = hutils.get_raven_loc()
sys.path.append(os.path.expanduser(raven_path))
from MessageHandler import MessageUser


class Base(MessageUser):
  def __init__(self, messageHandler=None, **kwargs):
    """
      Constructor.
      @ In, messageHandler, MessageHandler, message handling object
      @ In, kwargs, dict, passthrough arguments
      @ Out, None
    """
    self.messageHandler = messageHandler
    #if self.messageHandler is not None:
    #  self.raiseADebug('Set message handler for', self)

  def __repr__(self):
    """
      String representation.
      @ In, None
      @ Out, repr, str rep
    """
    return '<HERON {}>'.format(self.__class__.__name__)

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
    print('For {} no printing has been implemented!'.format(self))

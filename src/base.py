"""
  Base classes for other classes.
"""
import os
import sys
raven_path = '~/projects/raven/framework'
sys.path.append(os.path.expanduser(raven_path))
from MessageHandler import MessageUser


class Base(MessageUser):
  def __init__(self, messageHandler=None, **kwargs):
    self.messageHandler = messageHandler
    #if self.messageHandler is not None:
    #  self.raiseADebug('Set message handler for', self)

  def __repr__(self):
    return '<EGRET {}>'.format(self.__class__.__name__)

  def set_message_handler(self, mh):
    self.messageHandler = mh

  def print_me(self, **kwargs):
    print('For {} no printing has been implemented!'.format(self))

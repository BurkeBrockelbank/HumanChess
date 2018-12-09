"""
This program contains classes for building neural nets for HumanChess

Project: HumanChess
Path: root/net.py
"""

import torch
import chess

class BaseNet(nn.Module):
    """
    Base class for neural nets.
    """
    def __init__(self):
        pass
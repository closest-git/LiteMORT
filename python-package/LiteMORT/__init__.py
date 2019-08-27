# coding: utf-8
"""LiteMORT, Light Gradient Boosting Machine.

__author__ = 'Yingshi Chen'
"""
from __future__ import absolute_import
import os


try:
    #from .LiteMORT_problems import Mort_Problems
    from .LiteMORT import LiteMORT
    from .LiteMORT_preprocess import Mort_Preprocess,Mort_PickSamples
except ImportError:
    pass
'''
try:
    from .plotting import plot_importance, plot_metric, plot_tree, create_tree_digraph
except ImportError:
    pass
'''

dir_path = os.path.dirname(os.path.realpath(__file__))

if os.path.isfile(os.path.join(dir_path, 'VERSION.txt')):
    __version__ = open(os.path.join(dir_path, 'VERSION.txt')).read().strip()

__all__ = ['LiteMORT','Mort_Preprocess']



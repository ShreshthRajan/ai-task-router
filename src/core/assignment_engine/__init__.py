"""
Assignment Engine Package

This package contains the core assignment optimization algorithms for the AI Task Router.
It implements multi-objective optimization for intelligent task-developer matching.
"""

from .optimizer import AssignmentOptimizer
from .learning_automata import LearningAutomata

__all__ = ['AssignmentOptimizer', 'LearningAutomata']
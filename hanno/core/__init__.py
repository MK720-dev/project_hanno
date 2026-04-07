"""
Core package for shared Hanno infrastructure.

The core package holds small, dependency-light building blocks that are reused
across the rest of the framework. In the first implementation batch, this
includes shared dataclasses and reproducibility helpers.

The goal is to keep these utilities generic and stable so other modules can
depend on them without creating circular dependencies.
"""
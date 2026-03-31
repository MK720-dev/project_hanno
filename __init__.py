"""
Hanno package root.

This file marks the top-level package for the Hanno reinforcement-learning-based
optimization project. The initial implementation focuses on the restricted
proof-of-concept schema, where a controller observes optimization diagnostics
and modulates a fixed optimizer backbone.

Keeping this file explicit is useful even though it is small:
- it makes the package importable immediately,
- it gives the project a stable package root,
- and it provides a clear anchor point for later package-level exports.
"""
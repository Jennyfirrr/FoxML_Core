# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Target Routing Module

Routes target columns to training specifications.
"""

from .target_router import TaskSpec, route_target

__all__ = ['TaskSpec', 'route_target']


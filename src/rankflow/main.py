"""Backward-compatible re-export shim.

Preserves `from rankflow.main import RankFlow` for existing users.
"""

from rankflow.core import RankFlow

__all__ = ["RankFlow"]

"""GARVIS core package.

Load order matters: config -> logger -> safety -> memory -> permissions ->
brain. Nothing here imports `main` or `tools` at module import time, so the
package can be imported from tests without side effects.
"""

__all__ = ["config", "logger", "safety", "memory", "permissions", "brain", "state"]

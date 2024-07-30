try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

__author__ = "OneDiff"
__credits__ = "OneDiff contributors"

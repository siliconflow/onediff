from ._config import ipadapter_plus_hijacker, is_load_ipadapter_plus_pkg

if is_load_ipadapter_plus_pkg:
    from .IPAdapterPlus import *

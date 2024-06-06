from .hijack_samplers import samplers_hijack
from .hijack_ipadapter_plus import ipadapter_plus_hijacker

samplers_hijack.hijack(last=False)
ipadapter_plus_hijacker.hijack(last=False)

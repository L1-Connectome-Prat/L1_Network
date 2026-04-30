"""
Shared compartment palette used by network/compartment figures.

Order is chosen to group anatomically related compartments with similar hues:
    - MB-CA / MB / MBE : blue family, MB darkest
    - SMP / SMPal      : gold family
    - LAL-VMC          : RGB blend of LAL and VMC

"""
import numpy as np
from matplotlib.colors import to_rgb

#------------------------------------------------------------------------------
COMP_HEX = {
    "AL"      : "#e6194b",   # red
    "TR"      : "#f58231",   # orange
    "LON"     : "#17becf",   # cyan (larval optic neuropil)
    "MB-CA"   : "#6baed6",   # medium blue
    "MB"      : "#08306b",   # dark navy
    "MBE"     : "#9ecae1",   # light blue
    "IPA"     : "#9467bd",   # purple
    "IPLM"    : "#e377c2",   # pink-magenta
    "IPP"     : "#54278f",   # deep indigo-purple
    "SMP"     : "#bf812d",   # dark gold
    "SMPal"   : "#f7d488",   # light gold
    "SLP"     : "#01665e",   # dark teal
    "LAL"     : "#a50f15",   # burgundy
    "VMC"     : "#33a02c",   # vivid green
    "VLP"     : "#bcbd22",   # olive
    "SEZ-VNC" : "#34495e",   # slate
    "Mixed"   : "#a9a9a9",   # grey
}

# LAL-VMC is the RGB blend of LAL and VMC.
_lal = np.array(to_rgb(COMP_HEX["LAL"]))
_vmc = np.array(to_rgb(COMP_HEX["VMC"]))
COMP_RGB = {c : np.array(to_rgb(h)) for c, h in COMP_HEX.items()}
COMP_RGB["LAL-VMC"] = (_lal + _vmc) / 2

COMP_ORDER = ["AL", "TR", "LON",
              "MB-CA", "MB", "MBE",
              "IPA", "IPLM", "IPP",
              "SMP", "SMPal",
              "SLP",
              "LAL", "VMC", "LAL-VMC",
              "VLP",
              "SEZ-VNC",
              "Mixed"]
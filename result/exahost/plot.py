import matplotlib.pyplot as pl
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

# data path
dpth = "data/"

# output path
opth = "pdf/"

# yz size
DYZ = 256*256

# stencil size
stz = ["7p"]
"""
stz = ["7p", "13p", "19p", "25p", "31p", "37p"]
"""

# optimization
opt = ["Base"]
"""
opt = ["Base", 
        "Sham", 
        "ZintReg", 
        "Zint", 
        "ShamZintReg", 
        "ShamZint", 
        "ShamZintTempReg", 
        "Roc", 
        "ShamRoc", 
        "RocZintReg", 
        "RocZint", 
        "ShamRocZintTempReg"]
"""

for st in stz:
    for op in opt:
        file_name = st + "-" + op

        print "Reading " + file_name
        dt = pd.read_csv(dpth + file_name + ".out")

        fig, ax = pl.subplots(1)
        ax.plot(dt["dx"]*DYZ, dt["gFlops"])

        pp = PdfPages(opth + file_name + ".pdf")
        pp.savefig(fig)
        pp.close()

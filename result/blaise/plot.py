import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

# data path
dpth = "data/"

# output path
opth = "pdf/"

pltypes = [':','--','-']

def my_plot(ax, st, ymin, ymax, opts, output_file):
    ax.set_ylim(bottom=ymin,top=ymax)
    ax.xaxis.set_ticks(np.arange(32,1952+160,160))

    ax.set_title(st)
    ax.set_xlabel("X Axis Size")
    ax.set_ylabel("GFLOPs")

    for i, op in enumerate(opts):
        file_name = st + "-" + op + ".out"
        df = pd.read_csv(dpth + file_name)
        ax.plot(df["dx"], df["gFlops"], pltypes[i], label=op)

    ax.legend()


fig, axs = pl.subplots(nrows=3, ncols=2, figsize=(20,15))

fig.suptitle("Base-Sham-Roc")
my_plot(axs[0][0], "7p",  320, 500, ["Base","Sham","Roc"], "7p-Base-Sham-Roc")
my_plot(axs[0][1], "13p", 420, 850, ["Base","Sham","Roc"], "13p-Base-Sham-Roc")
my_plot(axs[1][0], "19p", 420, 1100, ["Base","Sham","Roc"], "19p-Base-Sham-Roc")
my_plot(axs[1][1], "25p", 450, 1250, ["Base","Sham","Roc"], "25p-Base-Sham-Roc")
my_plot(axs[2][0], "31p", 470, 1400, ["Base","Sham","Roc"], "31p-Base-Sham-Roc")
my_plot(axs[2][1], "37p", 470, 1500, ["Base","Sham","Roc"], "37p-Base-Sham-Roc")

pp = PdfPages(opth + "Base-Sham-Roc.pdf")
pp.savefig(fig)
pp.close()
pl.close(fig)

fig, axs = pl.subplots(nrows=3, ncols=2, figsize=(20,15))

fig.suptitle("Zint-ShamZint-RocZint")
my_plot(axs[0][0], "7p",  250, 550, ["Zint","ShamZint","RocZint"], "7p-Zint-ShamZint-RocZint")
my_plot(axs[0][1], "13p", 250, 700, ["Zint","ShamZint","RocZint"], "13p-Zint-ShamZint-RocZint")
my_plot(axs[1][0], "19p", 250, 1050, ["Zint","ShamZint","RocZint"], "19p-Zint-ShamZint-RocZint")
my_plot(axs[1][1], "25p", 250, 1050, ["Zint","ShamZint","RocZint"], "25p-Zint-ShamZint-RocZint")
my_plot(axs[2][0], "31p", 250, 1250, ["Zint","ShamZint","RocZint"], "31p-Zint-ShamZint-RocZint")
my_plot(axs[2][1], "37p", 250, 1200, ["Zint","ShamZint","RocZint"], "37p-Zint-ShamZint-RocZint")

pp = PdfPages(opth + "Zint-ShamZint-RocZint.pdf")
pp.savefig(fig)
pp.close()
pl.close(fig)

fig, axs = pl.subplots(nrows=3, ncols=2, figsize=(20,15))

fig.suptitle("ZintReg-ShamZintReg-RocZintReg")
my_plot(axs[0][0], "7p",  250, 700, ["ZintReg","ShamZintReg","RocZintReg"], "7p-ZintReg-ShamZintReg-RocZintReg")
my_plot(axs[0][1], "13p", 250, 1200, ["ZintReg","ShamZintReg","RocZintReg"], "7p-ZintReg-ShamZintReg-RocZintReg")
my_plot(axs[1][0], "19p", 250, 1600, ["ZintReg","ShamZintReg","RocZintReg"], "7p-ZintReg-ShamZintReg-RocZintReg")
my_plot(axs[1][1], "25p", 250, 2000, ["ZintReg","ShamZintReg","RocZintReg"], "7p-ZintReg-ShamZintReg-RocZintReg")
my_plot(axs[2][0], "31p", 250, 2400, ["ZintReg","ShamZintReg","RocZintReg"], "7p-ZintReg-ShamZintReg-RocZintReg")
my_plot(axs[2][1], "37p", 250, 2600, ["ZintReg","ShamZintReg","RocZintReg"], "7p-ZintReg-ShamZintReg-RocZintReg")

pp = PdfPages(opth + "ZintReg-ShamZintReg-RocZintReg.pdf")
pp.savefig(fig)
pp.close()
pl.close(fig)

fig, axs = pl.subplots(nrows=3, ncols=2, figsize=(20,15))

fig.suptitle("ShamRocZintTempReg-ShamZintTempReg")
my_plot(axs[0][0], "7p",  250, 900, ["ShamZintTempReg","ShamRocZintTempReg"], "7p-ShamZintTempReg-ShamRocZintTempReg")
my_plot(axs[0][1], "13p", 250, 1300, ["ShamZintTempReg","ShamRocZintTempReg"], "7p-ShamZintTempReg-ShamRocZintTempReg")
my_plot(axs[1][0], "19p", 250, 1300, ["ShamZintTempReg","ShamRocZintTempReg"], "7p-ShamZintTempReg-ShamRocZintTempReg")
my_plot(axs[1][1], "25p", 250, 1100, ["ShamZintTempReg","ShamRocZintTempReg"], "7p-ShamZintTempReg-ShamRocZintTempReg")
my_plot(axs[2][0], "31p", 250, 900, ["ShamZintTempReg","ShamRocZintTempReg"], "7p-ShamZintTempReg-ShamRocZintTempReg")
my_plot(axs[2][1], "37p", 250, 600, ["ShamZintTempReg","ShamRocZintTempReg"], "7p-ShamZintTempReg-ShamRocZintTempReg")

pp = PdfPages(opth + "ShamRocZintTempReg-ShamZintTempReg.pdf")
pp.savefig(fig)
pp.close()
pl.close(fig)

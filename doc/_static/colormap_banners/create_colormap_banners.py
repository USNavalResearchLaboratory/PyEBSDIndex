# Colormap banners used on the doc landing page. Run from this directory
# to place the banners in this directory, which is assumed in the docs.

import matplotlib as mpl
import matplotlib.pyplot as plt


plt.rcParams["backend"] = "agg"

cmaps = [mpl.cm.plasma, mpl.cm.viridis]

for i, cmap in enumerate(cmaps):
    fig, ax = plt.subplots(figsize=(30, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.Normalize(vmin=5, vmax=10)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )
    cbar.ax.set_axis_off()
    cbar.ax.margins(0, 0)
    fig.savefig(f"banner{i}.png", pad_inches=0, bbox_inches="tight")
    i += 1

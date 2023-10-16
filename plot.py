import abc, os, numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils import replace_chars


class Plot(abc.ABC):
    def __init__(self, fig_title: str, group_title: str, hparams_in_plot_title: bool = True, plot_title_fontsize: int = 18) -> None:
        self.fig, self.ax = plt.subplots(constrained_layout=True)
        # Format complete title
        plot_title = fig_title if not hparams_in_plot_title else fig_title + ' - ' + group_title
        self.ax.set_title(plot_title, fontsize=plot_title_fontsize)        
        # Remove LateX characters for figure title
        self.title = replace_chars(fig_title + ' - ' + group_title, '\n$\\', '')
        self.fig.canvas.manager.set_window_title(self.title)

    @abc.abstractmethod
    def addline(*args, **kwargs):
        ...

    def showlegend(self) -> None:
        ...

    def savefig(self, path: str):
        self.fig.savefig(os.path.join(path, self.title))


class AccuracyPlot(Plot):
    def __init__(self, fig_title: str, group_title: str, hparams_in_plot_title: bool = True, plot_title_fontsize: int = 18, 
                 y_lim_bottom: Optional[int] = None, y_lim_top: Optional[int] = None, set_label: Optional[tuple] = ('round', 'accuracy ($\%$)'), 
                 xticks: Optional[list] = None, yticks: Optional[list] = None):
        super().__init__(fig_title, group_title, hparams_in_plot_title, plot_title_fontsize)
        self.ax.grid(color='silver', linestyle='--', axis='both')

        if set_label:
            self.ax.set(xlabel=set_label[0], ylabel=set_label[1])
        if y_lim_bottom:
            self.ax.set_ylim(bottom=y_lim_bottom)
        if y_lim_top:
            self.ax.set_ylim(top=y_lim_top)
        self.xticks = xticks
        self.yticks = yticks

    def addline(self, label: str, data: dict, nbins: int = 6, plot_std: bool = False, 
                scale_x_axis: bool = False, scale_y_axis: bool = False, *args, **kwargs):
        accuracy = data['accuracy'] 
        self.ax.plot(accuracy.keys(), accuracy.values(), *args, label=label, linewidth=2, alpha=0.8, **kwargs)
        if plot_std:
            std = data['accuracy std']
            std_values = np.array([v for v in std.values()])
            accuracy_values = np.array([v for v in accuracy.values()])
            self.ax.fill_between(accuracy.keys(), accuracy_values - std_values, accuracy_values + std_values, *args, alpha=0.2, **kwargs)
        self._set_ticks(nbins)
        self._scale_axis(scale_x_axis, scale_y_axis)
        return self
    
    def _set_ticks(self, nbins: int):
        if not self.xticks:
            self.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins))
            self.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins))
        if self.xticks:
            self.ax.set_xticks(self.xticks)
        if self.yticks:
            self.ax.set_yticks(self.yticks)
    
    def _scale_axis(self, scale_x_axis: bool, scale_y_axis: bool) -> None:
        if scale_x_axis:
            self.ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))        
            self.ax.ticklabel_format(style="sci",  axis="x", scilimits=(0,0))
        if scale_y_axis:
            self.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))        
            self.ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))

    def showlegend(self) -> None:
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles, labels, loc='best', ncol=2)


    
class Heatmap(Plot):
    def __init__(self, fig_title: str, group_title: str, hparams_in_plot_title: bool, plot_title_fontsize: int, row_labels: List[str], 
                 col_labels: List[str], cbar_kw: Optional[dict] = None, cbarlabel: Optional[str] = None, **kwargs) -> None:
        super().__init__(fig_title, group_title, hparams_in_plot_title, plot_title_fontsize)
        self.fig.set_tight_layout(True)
        self.cbar_kw = cbar_kw or {}
        self.cbarlabel = cbarlabel

        self.row_labels = row_labels
        self.col_labels = col_labels
        self.imshowargs = kwargs
        self.acc = np.full((len(row_labels), len(col_labels),), np.nan)
        self.std = np.full((len(row_labels), len(col_labels),), np.nan)

    def addline(self, row_name: str, col_name: str, acc: float, std: float):
        assert row_name in self.row_labels and col_name in self.col_labels, "Row and/or col name outside axes of heatmap"
        r, c = self.row_labels.index(row_name), self.col_labels.index(col_name)
        self.acc[r, c] = acc
        self.std[r, c] = std

        # Show the heatmap only if this is the last value inserted
        if not np.isnan(self.acc).any():
            im, _ = self._plot()
            _ = self._annotate(im)
            # self.fig.tight_layout()


    def _plot(self):
        ax = self.ax
        im = ax.imshow(self.acc, **self.imshowargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **self.cbar_kw)
        cbar.ax.set_ylabel(self.cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(self.acc.shape[1]), labels=self.col_labels)
        ax.set_yticks(np.arange(self.acc.shape[0]), labels=self.row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
        #         rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(self.acc.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(self.acc.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        return im, cbar
    
    def _annotate(self, im, valfmt="${x[0]:.2f}_{{\\tiny\\pm {x[1]:.2f}}}$", textcolors=("black", "white"), 
                  threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """
        import matplotlib as plt
        data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                verticalalignment="center", fontsize=12, fontweight='roman')
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = plt.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt((data[i, j], self.std[i,j]), None), **kw)
                texts.append(text)

        return texts
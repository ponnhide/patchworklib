import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import product
from distutils.version import LooseVersion, StrictVersion

import seaborn 
from inspect import signature
from seaborn.axisgrid import FacetGrid, JointGrid, PairGrid, Grid 
import seaborn.matrix as sm
from seaborn.matrix import ClusterGrid
if (seaborn.__version__) >= StrictVersion("0.13"):  
    from seaborn._base import VectorPlotter, categorical_order
elif (seaborn.__version__) >= StrictVersion("0.12"):  
    from seaborn._oldcore import VectorPlotter, variable_type, categorical_order
else:
    from seaborn._core import VectorPlotter, variable_type, categorical_order

from seaborn import utils
from seaborn.utils import _check_argument, adjust_legend_subtitles, _draw_figure
from seaborn.palettes import color_palette, blend_palette
from seaborn._docstrings import (
    DocstringComponents,
    _core_docs,
)


#Referred to Seaborn, v0.11.2, Copyright (c) 2012-2021, Michael L. Waskom.
def add_legend(self, legend_data=None, title=None, label_order=None,
               adjust_subtitles=False, **kwargs):
    
    if legend_data is None:
        legend_data = self._legend_data
    if label_order is None:
        if self.hue_names is None:
            label_order = list(legend_data.keys())
        else:
            label_order = list(map(utils.to_utf8, self.hue_names))

    blank_handle = mpl.patches.Patch(alpha=0, linewidth=0)
    handles = [legend_data.get(l, blank_handle) for l in label_order]
    title = self._hue_var if title is None else title
    title_size = mpl.rcParams["legend.title_fontsize"]

    labels = []
    for entry in label_order:
        if isinstance(entry, tuple):
            _, label = entry
        else:
            label = entry
        labels.append(label)

    kwargs.setdefault("scatterpoints", 1)

    if self._legend_out:
        kwargs.setdefault("frameon", False)
        kwargs.setdefault("loc", "center right")
        self._figlegend = (handles, labels, kwargs, title, title_size)
         
    else:
        ax = self.axes.flat[0]
        kwargs.setdefault("loc", "best")

        leg = ax.legend(handles, labels, **kwargs)
        leg.set_title(title, prop={"size": title_size})
        self._legend = leg

        if adjust_subtitles:
            adjust_legend_subtitles(leg)

    return self

def __init_for_facetgrid__(
    self, data, *,
    row=None, col=None, hue=None, col_wrap=None,
    sharex=True, sharey=True, height=3, aspect=1, palette=None,
    row_order=None, col_order=None, hue_order=None, hue_kws=None,
    dropna=False, legend_out=True, despine=True,
    margin_titles=False, xlim=None, ylim=None, subplot_kws=None,
    gridspec_kws=None, size=None, pyplot=True,
):

    super(FacetGrid, self).__init__()
    
    if size is not None:
        height = size
        msg = ("The `size` parameter has been renamed to `height`; "
               "please update your code.")
        warnings.warn(msg, UserWarning)

    hue_var = hue
    if hue is None:
        hue_names = None
    else:
        hue_names = categorical_order(data[hue], hue_order)

    colors = self._get_palette(data, hue, hue_order, palette)

    if row is None:
        row_names = []
    else:
        row_names = categorical_order(data[row], row_order)

    if col is None:
        col_names = []
    else:
        col_names = categorical_order(data[col], col_order)

    hue_kws = hue_kws if hue_kws is not None else {}

    none_na = np.zeros(len(data), bool)
    if dropna:
        row_na = none_na if row is None else data[row].isnull()
        col_na = none_na if col is None else data[col].isnull()
        hue_na = none_na if hue is None else data[hue].isnull()
        not_na = ~(row_na | col_na | hue_na)
    else:
        not_na = ~none_na

    ncol = 1 if col is None else len(col_names)
    nrow = 1 if row is None else len(row_names)
    self._n_facets = ncol * nrow

    self._col_wrap = col_wrap
    if col_wrap is not None:
        if row is not None:
            err = "Cannot use `row` and `col_wrap` together."
            raise ValueError(err)
        ncol = col_wrap
        nrow = int(np.ceil(len(col_names) / col_wrap))
    self._ncol = ncol
    self._nrow = nrow

    figsize = (ncol * height * aspect, nrow * height)

    if col_wrap is not None:
        margin_titles = False

    subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
    gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
    if xlim is not None:
        subplot_kws["xlim"] = xlim
    if ylim is not None:
        subplot_kws["ylim"] = ylim

    fig = Grid._figure #Modified by Hideto
    self._figsize = figsize

    if col_wrap is None:

        kwargs = dict(squeeze=False,
                      sharex=sharex, sharey=sharey,
                      subplot_kw=subplot_kws,
                      gridspec_kw=gridspec_kws)

        axes = fig.subplots(nrow, ncol, **kwargs)

        if col is None and row is None:
            axes_dict = {}
        elif col is None:
            axes_dict = dict(zip(row_names, axes.flat))
        elif row is None:
            axes_dict = dict(zip(col_names, axes.flat))
        else:
            facet_product = product(row_names, col_names)
            axes_dict = dict(zip(facet_product, axes.flat))

    else:
        if gridspec_kws:
            warnings.warn("`gridspec_kws` ignored when using `col_wrap`")

        n_axes = len(col_names)
        axes = np.empty(n_axes, object)
        axes[0] = fig.add_subplot(nrow, ncol, 1, **subplot_kws)
        if sharex:
            subplot_kws["sharex"] = axes[0]
        if sharey:
            subplot_kws["sharey"] = axes[0]
        for i in range(1, n_axes):
            axes[i] = fig.add_subplot(nrow, ncol, i + 1, **subplot_kws)

        axes_dict = dict(zip(col_names, axes))

    self._figure = fig
    self._axes = axes
    self._axes_dict = axes_dict
    self._legend = None

    self.data = data
    self.row_names = row_names
    self.col_names = col_names
    self.hue_names = hue_names
    self.hue_kws = hue_kws

    self._nrow = nrow
    self._row_var = row
    self._ncol = ncol
    self._col_var = col

    self._margin_titles = margin_titles
    self._margin_titles_texts = []
    self._col_wrap = col_wrap
    self._hue_var = hue_var
    self._colors = colors
    self._legend_out = legend_out
    self._legend_data = {}
    self._x_var = None
    self._y_var = None
    self._sharex = sharex
    self._sharey = sharey
    self._dropna = dropna
    self._not_na = not_na

    self.set_titles()
    self.tight_layout()

    if despine:
        for ax in np.array(axes).flatten():
            ax.spines["right"].set_visible(False)   
            ax.spines["left"].set_visible(True) 
            ax.spines["top"].set_visible(False) 
            ax.spines["bottom"].set_visible(True) 

    if sharex in [True, 'col']:
        for ax in self._not_bottom_axes:
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)
            ax.xaxis.label.set_visible(False)

    if sharey in [True, 'row']:
        for ax in self._not_left_axes:
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.yaxis.label.set_visible(False)

def despine(self, right=True, left=False, top=True, bottom=False):
    if right == True:
        for axs in self._axes:
            for ax in axs:
                ax.spines["right"].set_visible(False)   

    if left == True:
        for axs in self._axes:
            for ax in axs:
                ax.spines["left"].set_visible(False)   

    if bottom == True:
        for axs in self._axes:
            for ax in axs:
                ax.spines["bottom"].set_visible(False)   
   
    if top == True:
        for axs in self._axes:
            for ax in axs:
                ax.spines["top"].set_visible(False)   

def __init_for_pairgrid__(
        self, data, *,
        hue=None, hue_order=None, palette=None,
        hue_kws=None, vars=None, x_vars=None, y_vars=None,
        corner=False, diag_sharey=True, height=2.5, aspect=1,
        layout_pad=.5, despine=True, dropna=False, size=None
    ):
    

    super(PairGrid, self).__init__()

    if size is not None:
        height = size
        msg = ("The `size` parameter has been renamed to `height`; "
               "please update your code.")
        warnings.warn(UserWarning(msg))

    numeric_cols = self._find_numeric_cols(data)
    if hue in numeric_cols:
        numeric_cols.remove(hue)
    if vars is not None:
        x_vars = list(vars)
        y_vars = list(vars)
    if x_vars is None:
        x_vars = numeric_cols
    if y_vars is None:
        y_vars = numeric_cols

    if np.isscalar(x_vars):
        x_vars = [x_vars]
    if np.isscalar(y_vars):
        y_vars = [y_vars]

    self.x_vars = x_vars = list(x_vars)
    self.y_vars = y_vars = list(y_vars)
    self.square_grid = self.x_vars == self.y_vars

    if not x_vars:
        raise ValueError("No variables found for grid columns.")
    if not y_vars:
        raise ValueError("No variables found for grid rows.")

    figsize = len(x_vars) * height * aspect, len(y_vars) * height

    fig = Grid._figure
    self._figsize = figsize
    axes = fig.subplots(len(y_vars), len(x_vars),
                        sharex="col", sharey="row",
                        squeeze=False)

    self._corner = corner
    if corner:
        hide_indices = np.triu_indices_from(axes, 1)
        for i, j in zip(*hide_indices):
            axes[i, j].remove()
            axes[i, j] = None

    self._figure = fig
    self.axes = axes
    self.data = data

    self.diag_sharey = diag_sharey
    self.diag_vars = None
    self.diag_axes = None

    self._dropna = dropna

    self._add_axis_labels()

    self._hue_var = hue
    if hue is None:
        self.hue_names = hue_order = ["_nolegend_"]
        self.hue_vals = pd.Series(["_nolegend_"] * len(data),
                                  index=data.index)
    else:
        hue_names = hue_order = categorical_order(data[hue], hue_order)
        if dropna:
            hue_names = list(filter(pd.notnull, hue_names))
        self.hue_names = hue_names
        self.hue_vals = data[hue]

    self.hue_kws = hue_kws if hue_kws is not None else {}

    self._orig_palette = palette
    self._hue_order = hue_order
    self.palette = self._get_palette(data, hue, hue_order, palette)
    self._legend_data = {}

    for ax in axes[:-1, :].flat:
        if ax is None:
            continue
        for label in ax.get_xticklabels():
            label.set_visible(False)
        ax.xaxis.offsetText.set_visible(False)
        ax.xaxis.label.set_visible(False)

    for ax in axes[:, 1:].flat:
        if ax is None:
            continue
        for label in ax.get_yticklabels():
            label.set_visible(False)
        ax.yaxis.offsetText.set_visible(False)
        ax.yaxis.label.set_visible(False)

    self._tight_layout_rect = [.01, .01, .99, .99]
    self._tight_layout_pad = layout_pad
    self._despine = despine
    if despine:
        for axs in axes:
            for ax in axs:
                ax.spines["right"].set_visible(False)   
                ax.spines["left"].set_visible(True) 
                ax.spines["top"].set_visible(False) 
                ax.spines["bottom"].set_visible(True) 
    
    self.tight_layout(pad=layout_pad)

def __init_for_jointgrid__(
    self, *,
    x=None, y=None,
    data=None,
    height=6, ratio=5, space=.2,
    dropna=False, xlim=None, ylim=None, size=None, marginal_ticks=False,
    hue=None, palette=None, hue_order=None, hue_norm=None,
):
    # Handle deprecations
    if size is not None:
        height = size
        msg = ("The `size` parameter has been renamed to `height`; "
               "please update your code.")
        warnings.warn(msg, UserWarning)

    f =  Grid._figure
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    ax_joint = f.add_subplot(gs[1:, :-1])
    ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

    self._figure  = f
    self._figsize = (height, height) 
    self.ax_joint = ax_joint
    self.ax_marg_x = ax_marg_x
    self.ax_marg_y = ax_marg_y

    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)

    if not marginal_ticks:
        plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_x.get_yticklabels(), visible=False)
        plt.setp(ax_marg_y.get_xticklabels(), visible=False)
        plt.setp(ax_marg_x.get_yticklabels(minor=True), visible=False)
        plt.setp(ax_marg_y.get_xticklabels(minor=True), visible=False)
        ax_marg_x.yaxis.grid(False)
        ax_marg_y.xaxis.grid(False)

    p = VectorPlotter(data=data, variables=dict(x=x, y=y, hue=hue))
    plot_data = p.plot_data.loc[:, p.plot_data.notna().any()]

    if dropna:
        plot_data = plot_data.dropna()

    def get_var(var):
        vector = plot_data.get(var, None)
        if vector is not None:
            vector = vector.rename(p.variables.get(var, None))
        return vector

    self.x = get_var("x")
    self.y = get_var("y")
    self.hue = get_var("hue")

    for axis in "xy":
        name = p.variables.get(axis, None)
        if name is not None:
            getattr(ax_joint, f"set_{axis}label")(name)

    if xlim is not None:
        ax_joint.set_xlim(xlim)
    if ylim is not None:
        ax_joint.set_ylim(ylim)

    self._hue_params = dict(palette=palette, hue_order=hue_order, hue_norm=hue_norm)

    ax_joint.spines["right"].set_visible(False)   
    ax_joint.spines["left"].set_visible(True) 
    ax_joint.spines["top"].set_visible(False) 
    ax_joint.spines["bottom"].set_visible(True) 
    if not marginal_ticks:
        utils.despine(ax=ax_marg_x, left=True)
        utils.despine(ax=ax_marg_y, bottom=True)
    for axes in [ax_marg_x, ax_marg_y]:
        for axis in [axes.xaxis, axes.yaxis]:
            axis.label.set_visible(False)
    f.tight_layout()
    #f.subplots_adjust(hspace=space, wspace=space)

def __setattr_for_clustergrid__(self, key, value):
    if key == "_figure":
        self.__dict__[key] = Grid._figure 
        self.__dict__["_figsize"] = (5,5)
    else:
        super.__setattr__(self, key, value)

def __init_for_clustergrid__(self, data, pivot_kws=None, z_score=None, standard_scale=None,
             figsize=None, row_colors=None, col_colors=None, mask=None,
             dendrogram_ratio=None, colors_ratio=None, cbar_pos=None):
    """Grid object for organizing clustered heatmap input on to axes"""
    try:
        import scipy
    except:
        raise RuntimeError("ClusterGrid requires scipy to be available")

    if isinstance(data, pd.DataFrame):
        self.data = data
    else:
        self.data = pd.DataFrame(data)

    self.data2d = self.format_data(self.data, pivot_kws, z_score,
                                   standard_scale)

    self.mask = sm._matrix_mask(self.data2d, mask)
    
    self._figure  = Grid._figure #Modified by Hideto
    self._figsize = figsize
    self._figure.set_size_inches(figsize)
    #self._figure = plt.figure(figsize=figsize)

    self.row_colors, self.row_color_labels = \
        self._preprocess_colors(data, row_colors, axis=0)
    self.col_colors, self.col_color_labels = \
        self._preprocess_colors(data, col_colors, axis=1)

    try:
        row_dendrogram_ratio, col_dendrogram_ratio = dendrogram_ratio
    except TypeError:
        row_dendrogram_ratio = col_dendrogram_ratio = dendrogram_ratio

    try:
        row_colors_ratio, col_colors_ratio = colors_ratio
    except TypeError:
        row_colors_ratio = col_colors_ratio = colors_ratio

    width_ratios = self.dim_ratios(self.row_colors,
                                   row_dendrogram_ratio,
                                   row_colors_ratio)
    height_ratios = self.dim_ratios(self.col_colors,
                                    col_dendrogram_ratio,
                                    col_colors_ratio)

    nrows = 2 if self.col_colors is None else 3
    ncols = 2 if self.row_colors is None else 3

    self.gs = gridspec.GridSpec(nrows, ncols,
                                width_ratios=width_ratios,
                                height_ratios=height_ratios, wspace=0, hspace=0)

    self.ax_row_dendrogram = self._figure.add_subplot(self.gs[-1, 0])
    self.ax_col_dendrogram = self._figure.add_subplot(self.gs[0, -1])
    self.ax_row_dendrogram.set_axis_off()
    self.ax_col_dendrogram.set_axis_off()

    self.ax_row_colors = None
    self.ax_col_colors = None

    if self.row_colors is not None:
        self.ax_row_colors = self._figure.add_subplot(
            self.gs[-1, 1])
    if self.col_colors is not None:
        self.ax_col_colors = self._figure.add_subplot(
            self.gs[1, -1])

    self.ax_heatmap = self._figure.add_subplot(self.gs[-1, -1])
    if cbar_pos is None:
        self.ax_cbar = self.cax = None
    else:
        # Initialize the colorbar axes in the gridspec so that tight_layout
        # works. We will move it where it belongs later. This is a hack.
        self.ax_cbar = self._figure.add_subplot(self.gs[0, 0])
        self.cax = self.ax_cbar  # Backwards compatibility
    self.cbar_pos = cbar_pos

    self.dendrogram_row = None
    self.dendrogram_col = None

def __plot_for_clustergrid__(self, metric, method, colorbar_kws, row_cluster, col_cluster, row_linkage, col_linkage, tree_kws, **kws):
    # heatmap square=True sets the aspect ratio on the axes, but that is
    # not compatible with the multi-axes layout of clustergrid
    if kws.get("square", False):
        msg = "``square=True`` ignored in clustermap"
        warnings.warn(msg)
        kws.pop("square")

    colorbar_kws = {} if colorbar_kws is None else colorbar_kws

    self.plot_dendrograms(row_cluster, col_cluster, metric, method,
                          row_linkage=row_linkage, col_linkage=col_linkage,
                          tree_kws=tree_kws)
    
    try:
        xind = self.dendrogram_col.reordered_ind
    except AttributeError:
        xind = np.arange(self.data2d.shape[1])
    
    try:
        yind = self.dendrogram_row.reordered_ind
    except AttributeError:
        yind = np.arange(self.data2d.shape[0])

    self.plot_colors(xind, yind, **kws)
    self.plot_matrix(colorbar_kws, xind, yind, **kws)
    self._figure.set_size_inches((1,1))
    #mpl.rcParams["figure.subplot.hspace"] = hspace 
    #mpl.rcParams["figure.subplot.wspace"] = wspace
    return self


import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product

import seaborn 
from seaborn.axisgrid import FacetGrid, JointGrid, PairGrid, Grid 
from seaborn._core import VectorPlotter, variable_type, categorical_order
from seaborn import utils
from seaborn.utils import _check_argument, adjust_legend_subtitles, _draw_figure
from seaborn.palettes import color_palette, blend_palette
from seaborn._decorators import _deprecate_positional_args
from seaborn._docstrings import (
    DocstringComponents,
    _core_docs,
)

def add_legend(self, legend_data=None, title=None, label_order=None,
               adjust_subtitles=False, **kwargs):
    """Draw a legend, maybe placing it outside axes and resizing the figure.
    Parameters
    ----------
    legend_data : dict
        Dictionary mapping label names (or two-element tuples where the
        second element is a label name) to matplotlib artist handles. The
        default reads from ``self._legend_data``.
    title : string
        Title for the legend. The default reads from ``self._hue_var``.
    label_order : list of labels
        The order that the legend entries should appear in. The default
        reads from ``self.hue_names``.
    adjust_subtitles : bool
        If True, modify entries with invisible artists to left-align
        the labels and set the font size to that of a title.
    kwargs : key, value pairings
        Other keyword arguments are passed to the underlying legend methods
        on the Figure or Axes object.
    Returns
    -------
    self : Grid instance
        Returns self for easy chaining.
    """
    # Find the data for the legend
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

    # Unpack nested labels from a hierarchical legend
    labels = []
    for entry in label_order:
        if isinstance(entry, tuple):
            _, label = entry
        else:
            label = entry
        labels.append(label)

    # Set default legend kwargs
    kwargs.setdefault("scatterpoints", 1)

    if self._legend_out:
        kwargs.setdefault("frameon", False)
        kwargs.setdefault("loc", "center right")
        self._figlegend = (handles, labels, kwargs, title, title_size)
        
        # Draw a full-figure legend outside the grid
        #figlegend = self._figure.legend(handles, labels, **kwargs)
        
        #figlegend.set_title(title, prop={"size": title_size})

        #if adjust_subtitles:
        #    adjust_legend_subtitles(figlegend)
        
        # Draw the plot to set the bounding boxes correctly
        #_draw_figure(self._figure)

        # Calculate and set the new width of the figure so the legend fits
        #legend_width = figlegend.get_window_extent().width / self._figure.dpi
        #fig_width, fig_height = self._figure.get_size_inches()
        #self._figure.set_size_inches(fig_width + legend_width, fig_height)

        # Draw the plot again to get the new transformations
        #_draw_figure(self._figure)

        # Now calculate how much space we need on the right side
        #legend_width = figlegend.get_window_extent().width / self._figure.dpi
        #space_needed = legend_width / (fig_width + legend_width)
        #margin = .04 if self._margin_titles else .01
        #self._space_needed = margin + space_needed
        #right = 1 - self._space_needed

        # Place the subplot axes to give space for the legend
        #self._figure.subplots_adjust(right=right)
        #self._tight_layout_rect[2] = right
    
    else:
        # Draw a legend in the first axis
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
    
    # Handle deprecations
    if size is not None:
        height = size
        msg = ("The `size` parameter has been renamed to `height`; "
               "please update your code.")
        warnings.warn(msg, UserWarning)

    # Determine the hue facet layer information
    hue_var = hue
    if hue is None:
        hue_names = None
    else:
        hue_names = categorical_order(data[hue], hue_order)

    colors = self._get_palette(data, hue, hue_order, palette)

    # Set up the lists of names for the row and column facet variables
    if row is None:
        row_names = []
    else:
        row_names = categorical_order(data[row], row_order)

    if col is None:
        col_names = []
    else:
        col_names = categorical_order(data[col], col_order)

    # Additional dict of kwarg -> list of values for mapping the hue var
    hue_kws = hue_kws if hue_kws is not None else {}

    # Make a boolean mask that is True anywhere there is an NA
    # value in one of the faceting variables, but only if dropna is True
    none_na = np.zeros(len(data), bool)
    if dropna:
        row_na = none_na if row is None else data[row].isnull()
        col_na = none_na if col is None else data[col].isnull()
        hue_na = none_na if hue is None else data[hue].isnull()
        not_na = ~(row_na | col_na | hue_na)
    else:
        not_na = ~none_na

    # Compute the grid shape
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

    # Calculate the base figure size
    # This can get stretched later by a legend
    # TODO this doesn't account for axis labels
    figsize = (ncol * height * aspect, nrow * height)

    # Validate some inputs
    if col_wrap is not None:
        margin_titles = False

    # Build the subplot keyword dictionary
    subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
    gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
    if xlim is not None:
        subplot_kws["xlim"] = xlim
    if ylim is not None:
        subplot_kws["ylim"] = ylim

    # --- Initialize the subplot grid

    # Disable autolayout so legend_out works properly
    #with mpl.rc_context({"figure.autolayout": False}):
    #fig = mpl.figure.Figure(figsize=figsize)
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

        # If wrapping the col variable we need to make the grid ourselves
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

    # --- Set up the class attributes

    # Attributes that are part of the public API but accessed through
    # a  property so that Sphinx adds them to the auto class doc
    self._figure = fig
    self._axes = axes
    self._axes_dict = axes_dict
    self._legend = None

    # Public attributes that aren't explicitly documented
    # (It's not obvious that having them be public was a good idea)
    self.data = data
    self.row_names = row_names
    self.col_names = col_names
    self.hue_names = hue_names
    self.hue_kws = hue_kws

    # Next the private variables
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

    # --- Make the axes look good

    self.set_titles()
    self.tight_layout()

    if despine:
        #self.despine()
        for axs in axes:
            for ax in axs:
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
    
    """
    Initialize the plot figure and PairGrid object.
    Parameters
    ----------
    data : DataFrame
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
    hue : string (variable name)
        Variable in ``data`` to map plot aspects to different colors. This
        variable will be excluded from the default x and y variables.
    hue_order : list of strings
        Order for the levels of the hue variable in the palette
    palette : dict or seaborn color palette
        Set of colors for mapping the ``hue`` variable. If a dict, keys
        should be values  in the ``hue`` variable.
    hue_kws : dictionary of param -> list of values mapping
        Other keyword arguments to insert into the plotting call to let
        other plot attributes vary across levels of the hue variable (e.g.
        the markers in a scatterplot).
    vars : list of variable names
        Variables within ``data`` to use, otherwise use every column with
        a numeric datatype.
    {x, y}_vars : lists of variable names
        Variables within ``data`` to use separately for the rows and
        columns of the figure; i.e. to make a non-square plot.
    corner : bool
        If True, don't add axes to the upper (off-diagonal) triangle of the
        grid, making this a "corner" plot.
    height : scalar
        Height (in inches) of each facet.
    aspect : scalar
        Aspect * height gives the width (in inches) of each facet.
    layout_pad : scalar
        Padding between axes; passed to ``fig.tight_layout``.
    despine : boolean
        Remove the top and right spines from the plots.
    dropna : boolean
        Drop missing values from the data before plotting.
    See Also
    --------
    pairplot : Easily drawing common uses of :class:`PairGrid`.
    FacetGrid : Subplot grid for plotting conditional relationships.
    Examples
    --------
    .. include:: ../docstrings/PairGrid.rst
    """

    super(PairGrid, self).__init__()

    # Handle deprecations
    if size is not None:
        height = size
        msg = ("The `size` parameter has been renamed to `height`; "
               "please update your code.")
        warnings.warn(UserWarning(msg))

    # Sort out the variables that define the grid
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

    # Create the figure and the array of subplots
    figsize = len(x_vars) * height * aspect, len(y_vars) * height

    fig = Grid._figure
    self._figsize = figsize
    # Disable autolayout so legend_out works
    #with mpl.rc_context({"figure.autolayout": False}):
    #    fig = plt.figure(figsize=figsize)

    axes = fig.subplots(len(y_vars), len(x_vars),
                        sharex="col", sharey="row",
                        squeeze=False)

    # Possibly remove upper axes to make a corner grid
    # Note: setting up the axes is usually the most time-intensive part
    # of using the PairGrid. We are foregoing the speed improvement that
    # we would get by just not setting up the hidden axes so that we can
    # avoid implementing fig.subplots ourselves. But worth thinking about.
    self._corner = corner
    if corner:
        hide_indices = np.triu_indices_from(axes, 1)
        for i, j in zip(*hide_indices):
            axes[i, j].remove()
            axes[i, j] = None

    self._figure = fig
    self.axes = axes
    self.data = data

    # Save what we are going to do with the diagonal
    self.diag_sharey = diag_sharey
    self.diag_vars = None
    self.diag_axes = None

    self._dropna = dropna

    # Label the axes
    self._add_axis_labels()

    # Sort out the hue variable
    self._hue_var = hue
    if hue is None:
        self.hue_names = hue_order = ["_nolegend_"]
        self.hue_vals = pd.Series(["_nolegend_"] * len(data),
                                  index=data.index)
    else:
        # We need hue_order and hue_names because the former is used to control
        # the order of drawing and the latter is used to control the order of
        # the legend. hue_names can become string-typed while hue_order must
        # retain the type of the input data. This is messy but results from
        # the fact that PairGrid can implement the hue-mapping logic itself
        # (and was originally written exclusively that way) but now can delegate
        # to the axes-level functions, while always handling legend creation.
        # See GH2307
        hue_names = hue_order = categorical_order(data[hue], hue_order)
        if dropna:
            # Filter NA from the list of unique hue names
            hue_names = list(filter(pd.notnull, hue_names))
        self.hue_names = hue_names
        self.hue_vals = data[hue]

    # Additional dict of kwarg -> list of values for mapping the hue var
    self.hue_kws = hue_kws if hue_kws is not None else {}

    self._orig_palette = palette
    self._hue_order = hue_order
    self.palette = self._get_palette(data, hue, hue_order, palette)
    self._legend_data = {}

    # Make the plot look nice
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
    
    #utils.despine(fig=fig)
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

    # Set up the subplot grid
    #f = plt.figure(figsize=(height, height))
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

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
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

    # Process the input variables
    p = VectorPlotter(data=data, variables=dict(x=x, y=y, hue=hue))
    plot_data = p.plot_data.loc[:, p.plot_data.notna().any()]

    # Possibly drop NA
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

    # Store the semantic mapping parameters for axes-level functions
    self._hue_params = dict(palette=palette, hue_order=hue_order, hue_norm=hue_norm)

    # Make the grid look nice
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

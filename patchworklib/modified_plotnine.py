import numpy as np 
import itertools
from plotnine.facets.strips import Strips
from copy import deepcopy

def make_figure(self, figure=None):
    """
    Create and return Matplotlib figure and subplot axes
    """
    num_panels = len(self.layout.layout)
    axsarr = np.empty((self.nrow, self.ncol), dtype=object)

    # Create figure & gridspec
    if figure is None:
        figure, gs = self._make_figure()
    else:
        _, gs = self._make_figure()
    self.grid_spec = gs

    # Create axes
    it = itertools.product(range(self.nrow), range(self.ncol))
    for i, (row, col) in enumerate(it):
        axsarr[row, col] = figure.add_subplot(gs[i])

    # Rearrange axes
    # They are ordered to match the positions in the layout table
    if self.dir == "h":
        order: Literal["C", "F"] = "C"
        if not self.as_table:
            axsarr = axsarr[::-1]
    elif self.dir == "v":
        order = "F"
        if not self.as_table:
            axsarr = np.array([row[::-1] for row in axsarr])
    else:
        raise ValueError(f'Bad value `dir="{self.dir}"` for direction')

    axs = axsarr.ravel(order)

    # Delete unused axes
    for ax in axs[num_panels:]:
        figure.delaxes(ax)
    axs = axs[:num_panels]
    return figure, list(axs)

def setup(self, figure, plot):
    self.plot = plot
    self.layout = plot.layout

    if hasattr(plot, "figure"):
        self.figure, self.axs = plot.figure, plot.axs
    else:
        self.figure, self.axs = self.make_figure(self, figure=figure)

    self.coordinates = plot.coordinates
    self.theme = plot.theme
    self.layout.axs = self.axs
    self.strips = Strips.from_facet(self)
    return self.figure, self.axs

def newdraw(self, return_ggplot=False, show: bool = False):
    """
    Render the complete plot

    Parameters
    ----------
    show :
        Whether to show the plot.

    Returns
    -------
    :
        Matplotlib figure
    """
    import matplotlib as mpl 
    from plotnine._mpl.layout_engine import PlotnineLayoutEngine
    from plotnine.ggplot import plot_context

    # Do not draw if drawn already.
    # This prevents a needless error when reusing
    # figure & axes in the jupyter notebook.
    if hasattr(self, "figure"):
        return self.figure

    # Prevent against any modifications to the users
    # ggplot object. Do the copy here as we may/may not
    # assign a default theme
    self = deepcopy(self)
    with plot_context(self, show=show):
        self._build()

        # setup
        self.figure, self.axs = self.facet.setup(self)
        try:
            self.guides._setup(self)
        except:
            pass 

        self.theme.setup(self)

        # Drawing
        self._draw_layers()
        self._draw_panel_borders()
        self._draw_breaks_and_labels()
        self.guides.draw()
        self._draw_figure_texts()
        self._draw_watermarks()

        # Artist object theming
        self.theme.apply()
        self.figure.set_layout_engine(PlotnineLayoutEngine(self))
    if return_ggplot == True:
        return self.figure, self 
    else:
        return self.figure 
    return self.figure


def draw(self, return_ggplot=False, show: bool = False):
    """
    Render the complete plot

    Parameters
    ----------
    show : bool (default: False)
        Whether to show the plot.

    Returns
    -------
    fig : ~matplotlib.figure.Figure
        Matplotlib figure
    """
    import matplotlib as mpl 
    from plotnine._mpl.layout_engine import PlotnineLayoutEngine
    from plotnine.ggplot import plot_context
    plot_context.__exit__ = __exit__
    plot_context.__enter__ = __enter__
    # Do not draw if drawn already.
    # This prevents a needless error when reusing
    # figure & axes in the jupyter notebook.
    if hasattr(self, "figure"):
        return self.figure

    # Prevent against any modifications to the users
    # ggplot object. Do the copy here as we may/may not
    # assign a default theme
    self = deepcopy(self)
    
    with plot_context(self, show=show):
        self._build()

        # setup
        #figure, axs = self._create_figure()
        import matplotlib.pyplot as plt
        figure = plt.figure()
        axs = self.facet.make_axes(
            figure, self.layout.layout, self.coordinates
        )
        self.figure = figure 
        self.axs = axs
        self._setup_parameters()
        self.theme.setup()
        self.facet.strips.generate()

        # Drawing
        self._draw_layers()
        self._draw_breaks_and_labels()
        self._draw_legend()
        self._draw_figure_texts()
        self._draw_watermarks()

        # Artist object theming
        self.theme.apply()  
        figure.set_layout_engine(PlotnineLayoutEngine(self))

    if return_ggplot == True:
        return figure, self 
    else:
        return figure 

def __enter__(self):
    """
    Enclose in matplolib & pandas environments
    """
    import pandas as pd 
    import matplotlib as mpl

    self.plot.theme._targets = {}
    self.rc_context = mpl.rc_context(self.plot.theme.rcParams)
    # Pandas deprecated is_copy, and when we create new dataframes
    # from slices we do not want complaints. We always uses the
    # new frames knowing that they are separate from the original.
    self.pd_option_context = pd.option_context(
        "mode.chained_assignment", None
    )
    self.rc_context.__enter__()
    self.pd_option_context.__enter__()
    return self

def __exit__(self, exc_type, exc_value, exc_traceback):
    """
    Exit matplotlib & pandas environments
    """
    import matplotlib.pyplot as plt

    if exc_type is None:
        if self.show:
            plt.show()
        else:
            plt.close(self.plot.figure)
    else:
        # There is an exception, close any figure
        if hasattr(self.plot, "figure"):
            plt.close(self.plot.figure)

    self.rc_context.__exit__(exc_type, exc_value, exc_traceback)
    self.pd_option_context.__exit__(exc_type, exc_value, exc_traceback)

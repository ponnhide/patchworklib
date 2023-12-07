from copy import deepcopy

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

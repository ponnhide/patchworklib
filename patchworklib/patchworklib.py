import os
import matplotlib
import io
import sys 
import copy
import types
import dill 
import pickle
import warnings 
from math import log10 , floor
from distutils.version import LooseVersion, StrictVersion

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt  
import matplotlib.axes as axes
import plotnine

from contextlib import suppress
from types import SimpleNamespace as NS
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D

try:
    import patchworklib.modified_plotnine as mp9
    import patchworklib.modified_grid as mg
    import seaborn as sns
except Exception as e:
    print(e) 

#warnings.simplefilter('ignore', SettingWithCopyWarning)
warnings.simplefilter('ignore')

__version__     = "0.6.4" 
_basefigure     = plt.figure(figsize=(1,1))
_render         = _basefigure.canvas.get_renderer()
_scale          = Affine2D().scale(1./_basefigure.dpi)
_axes_dict      = {}
_removed_axes   = {}
_bricks_list    = [] 
basefigure      = _basefigure
axes_dict       = _axes_dict 

#default setting
matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.rcParams['ps.fonttype']       = 42
matplotlib.rcParams['pdf.fonttype']      = 42
matplotlib.rcParams['font.sans-serif']   = ["Arial","Lucida Sans","DejaVu Sans","Lucida Grande","Verdana"]
matplotlib.rcParams['font.family']       = 'sans-serif'
matplotlib.rcParams['font.size']         = 12.0
matplotlib.rcParams["axes.labelcolor"]   = "#000000"
matplotlib.rcParams["axes.linewidth"]    = 0.8
matplotlib.rcParams["xtick.major.width"] = 0.8
matplotlib.rcParams["xtick.minor.width"] = 0.4
matplotlib.rcParams["ytick.major.width"] = 0.8
matplotlib.rcParams["ytick.minor.width"] = 0.4
matplotlib.rcParams['xtick.major.pad']   = 4
matplotlib.rcParams['ytick.major.pad']   = 4
matplotlib.rcParams['xtick.major.size']  = 4
matplotlib.rcParams['ytick.major.size']  = 4
param = {"margin":0.5, "dpi":200}

def _round_it(x, sig):
    if x != 0:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    else:
        return x

def expand(bricks, w, h):
    """

    Expand the size of the bricks object.

    Parameters
    ----------
    bricks : patchworklib.Bricks object  
        A Bricks object to be expand.
    w : int or float  
        Expansion ratio of the width of an axes object.
    h : int or float  
        Expansion ratio of the width of an axes object.
    
    Returns
    -------
    None
    
    """

    global _axes_dict 
    x0, x1, y0, y1 = bricks.get_inner_corner()
    for key in bricks.bricks_dict:
        pos = bricks.bricks_dict[key].get_position()
        px0 = pos.x0 - x0
        px1 = pos.x1 - x0
        py0 = pos.y0 - y0 
        py1 = pos.y1 - y0
        ax = bricks.bricks_dict[key]
        ax.set_position([px0 * w, py0 * h, (px1-px0) * w, (py1-py0) * h])
        _reset_ggplot_legend(ax)
    
    for caselabel in bricks._case_labels:
        caselabel = caselabel[5:] 
        _reset_ggplot_legend(_axes_dict[caselabel])
    
    if bricks._label[0:15] == "Bricks-outline:" and w > 1 and h > 1:
        keys     = bricks.bricks_dict.keys() 
        new_dict    = {}
        case_labels = bricks._case_labels
        case_labels.remove("case:"+bricks._label) 
        for key in keys:
            if key[0:8] == "outline:":
                pass
            else:
                new_dict[key] = bricks.bricks_dict[key]
        
        tmp = Bricks(bricks_dict=new_dict)
        tmp._case_labels = case_labels
        posi_x0, posi_x1, posi_y0, posi_y1 = tmp.get_outer_corner()
        poso_x0, poso_x1, poso_y0, poso_y1 = bricks.bricks_dict["outline:" + bricks._label[15:]].get_outer_corner()
        
        w = abs(poso_x0 - poso_x1) / abs(posi_x0 - posi_x1) 
        h = abs(poso_y0 - poso_y1) / abs(posi_y0 - posi_y1) 
        x0, x1, y0, y1 = tmp.get_inner_corner()
        for key in tmp.bricks_dict:  
            pos = bricks.bricks_dict[key].get_position()
            px0 = pos.x0 - x0
            px1 = pos.x1 - x0
            py0 = pos.y0 - y0 
            py1 = pos.y1 - y0
            ax = bricks.bricks_dict[key]
            ax.set_position([px0 * w, py0 * h, (px1-px0) * w, (py1-py0) * h])
            _reset_ggplot_legend(ax)
        
        posi_x0, posi_x1, posi_y0, posi_y1 = tmp.get_outer_corner()
        dw = abs(poso_x1 - posi_x1)
        dh = abs(poso_y1 - posi_y1)
        
        x0, x1, y0, y1 = tmp.get_inner_corner()
        for key in tmp.bricks_dict:  
            pos = bricks.bricks_dict[key].get_position()
            px0 = pos.x0 - x0
            px1 = pos.x1 - x0
            py0 = pos.y0 - y0 
            py1 = pos.y1 - y0
            ax = bricks.bricks_dict[key]
            ax.set_position([px0 + dw, py0 + dh, (px1-px0), (py1-py0)])
            _reset_ggplot_legend(ax)

        for caselabel in bricks._case_labels:
            caselabel = caselabel[5:] 
            _reset_ggplot_legend(_axes_dict[caselabel])
    
    bricks.case 
    return bricks 

def _reset_ggplot_legend(bricks):
    """

    Reset location of the legend related with a ggplot or seaborn plot.

    Note
    ----
    The functions is a internal function. 

    Parameters
    ----------
    bricks : patchworklib.Bricks object
    
    Returns
    -------
    None.
    
    """

    if "_ggplot_legend" in bricks.__dict__ and bricks._ggplot_legend is not None:
        if (matplotlib.__version__) >= StrictVersion("3.7"):  
            for tmp_artist in bricks._case.artists:
                if tmp_artist == bricks._ggplot_legend:
                    tmp_artist.remove() 
        else:
            bricks._case.artists.remove(bricks._ggplot_legend)
        
        anchored_box = AnchoredOffsetbox(
            loc=bricks._ggplot_legend_loc,
            child=bricks._ggplot_legend_box,
            pad=0.,
            frameon=False,
            bbox_to_anchor=(bricks._ggplot_legend_x,bricks._ggplot_legend_y),
            bbox_transform = bricks._case.transAxes,
            borderpad=0.)
        anchored_box.set_zorder(90.1)
        anchored_box.set_in_layout(True)
        try:
            bricks._case.add_artist(anchored_box)
        except Exception as e:
            pass 
        bricks._ggplot_legend = anchored_box
        bricks.case
    
    if "_seaborn_legend" in bricks.__dict__ and bricks._seaborn_legend is not None:
        old_legend = bricks._case.legend_
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        if "bbox_to_anchor" in bricks._seaborn_legend[0]:
            del bricks._seaborn_legend[0]["bbox_to_anchor"] 
        bricks._case.legend(handles, labels, **bricks._seaborn_legend[0], title=title, bbox_to_anchor=bricks._seaborn_legend[1])

    else:
        pass

def overwrite_plotnine():
    plotnine.ggplot.draw = mp9.draw

def load_ggplot(ggplot=None, figsize=None):  
    """

    Convert a plotnine plot object to a patchworklib.Bricks object.

    Parameters
    ----------
    ggplot : plotnine.ggplot.ggplot
    figsize : tuple (float, float), 
        figure size. If it is not specified, convert Bricks object would keep the 
        original figure size of the given ggplot objecrt.
    
    Returns
    -------
    patchworklib.Bricks object. 
    
    """

    def draw_labels(bricks, gori, gcp, figsize):
        get_property = gcp.theme.themeables.property
        va = "top"
        try:
            margin = get_property('axis_title_x', 'margin')
        except KeyError:
            pad_x = 4
        else:
            if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
                pad_x = 14 + (get_property('axis_text_x', 'size') - 11) * 0.5 + (get_property('axis_title_x', 'size') - 11) * 0.5
            else:
                pad_x = margin.get_as('t', 'pt')

        try:
            margin = get_property('axis_title_y', 'margin')
        except KeyError:
            pad_y = 4
        else:
            if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
                pad_y = 12 + (get_property('axis_text_y', 'size') - 11) * 0.5 + (get_property('axis_title_y', 'size') - 11) * 0.5
            else:
                pad_y = margin.get_as('r', 'pt')

        labels = gcp.coordinates.labels(NS(
            x=gcp.layout.xlabel(gcp.labels),
            y=gcp.layout.ylabel(gcp.labels)
        ))
       
        if bricks._type == "Bricks":
            ix0, ix1, iy0, iy1 = bricks.get_inner_corner() 
            ox0, ox1, oy0, oy1 = bricks.get_outer_corner()
            
            px1 = (ix0 - ox0) / (ox1 - ox0)  
            px2 = (ix1 - ox0) / (ox1 - ox0) 
            xlabel = bricks.case.set_xlabel(labels.x, labelpad=pad_x, va=va)
            x,y = xlabel.get_position()
            xlabel.set_position([(px1+px2) / 2, y]) 

            py1 = (iy0 - oy0) / (oy1 - oy0)  
            py2 = (iy1 - oy0) / (oy1 - oy0) 
            ylabel = bricks.case.set_ylabel(labels.y, labelpad=pad_y)
            x,y = ylabel.get_position()
            ylabel.set_position([x, (py1+py2) / 2]) 

        else:
            xlabel = bricks.set_xlabel(labels.x, labelpad=pad_x, va="top")
            ylabel = bricks.set_ylabel(labels.y, labelpad=pad_y)    

        if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
            gori.theme._targets['axis_title_x'] = xlabel
            gori.theme._targets['axis_title_y'] = ylabel
            if 'axis_title_x' in gori.theme.themeables:
                gori.theme.themeables['axis_title_x'].apply_figure(gori.figure, gori.theme._targets)
                for ax in gori.axs:
                    gori.theme.themeables['axis_title_x'].apply_ax(ax)

            if 'axis_title_y' in gori.theme.themeables:
                gori.theme.themeables['axis_title_y'].apply_figure(gori.figure, gori.theme._targets)
                for ax in gori.axs:
                    gori.theme.themeables['axis_title_y'].apply_ax(ax)

            for key in gori.theme.themeables:
                if "legend" in key:
                    gori.theme.themeables[key].apply_figure(gori.figure, gori.theme._targets)
                    for ax in gori.axs:
                        gori.theme.themeables[key].apply_ax(ax) 
           
        else:
            gori.figure._themeable['axis_title_x'] = xlabel
            gori.figure._themeable['axis_title_y'] = ylabel
            if 'axis_title_x' in gori.theme.themeables:
                gori.theme.themeables['axis_title_x'].apply_figure(gori.figure)
                for ax in gori.axs:
                    gori.theme.themeables['axis_title_x'].apply(ax)

            if 'axis_title_y' in gori.theme.themeables:
                gori.theme.themeables['axis_title_y'].apply_figure(gori.figure)
                for ax in gori.axs:
                    gori.theme.themeables['axis_title_y'].apply(ax)
        
        return labels.x, labels.y

    def draw_legend(bricks, gori, gcp, figsize):
        get_property = gcp.theme.themeables.property
        legend_box   = gcp.guides.build(gcp)
        if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
            wratio = 1
            hratio = 1
        else:
            wratio  = figsize[0]
            hratio = figsize[1]
        
        try:
            spacing = get_property('legend_box_spacing')
        except KeyError:
            spacing = 0.1
        
        
        position = gcp.guides.position
        if position == 'right':
            loc = 6
            x = 1.0 + spacing/wratio
            y = 0.5
        elif position == 'left':
            loc = 7
            x = 0.0 - spacing/hratio
            y = 0.5
        elif position == 'top':
            loc = 8
            x = 0.5
            y = 1.0 + spacing/wratio
        elif position == 'bottom':
            loc = 9
            x = 0.5
            y = 0.0- spacing/hratio
        elif type(position) == tuple:
            loc = "center"
            x,y = position

        else:
            loc = 1
            x, y = 0, 0 
        
        if legend_box is None:
            pass 
        else:
            anchored_box = AnchoredOffsetbox(
                    loc=loc,
                    child=legend_box,
                    pad=0.,
                    frameon=False,
                    bbox_to_anchor=(x,y),
                    bbox_transform = bricks.case.transAxes,
                    borderpad=0.)
            
            anchored_box.set_zorder(90.1)
            anchored_box.set_in_layout(True)
            bricks.case.add_artist(anchored_box)
            bricks._ggplot_legend     = anchored_box
            bricks._ggplot_legend_box = legend_box  
            bricks._ggplot_legend_loc = loc
            bricks._ggplot_legend_x   = x
            bricks._ggplot_legend_y   = y
            if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
                gori.theme._targets['legend_background'] = anchored_box
                for key in gori.theme.themeables:
                    if "legend" in key:
                        gori.theme.themeables[key].apply_figure(gori.figure, gori.theme._targets)
                        for ax in gori.axs:
                            gori.theme.themeables[key].apply_ax(ax) 
            else:
                gori.figure._themeable['legend_background'] = anchored_box
                for key in gori.theme.themeables:
                    if "legend" in key:
                        gori.theme.themeables[key].apply_figure(gori.figure)
                        for ax in gori.axs:
                            gori.theme.themeables[key].apply(ax)

    def draw_title(bricks, gori, gcp, figsize):
        title = gcp.labels.get('title', '')
        rcParams = gcp.theme.rcParams
        get_property = gcp.theme.themeables.property
        try:
            fontsize = get_property('plot_title', 'size')
        except KeyError:
            fontsize = float(rcParams.get('font.size', 12))
        
        try:
            ha = get_property('plot_title', 'ha')
            mx0, mx1, my0, my1 = bricks.get_outer_corner() 
            ix0, ix1, iy0, iy1 = bricks.get_inner_corner() 

            if ha == "left":
                x = (ix0-mx0)/abs(mx1-mx0) 
                
            elif ha == "right": 
                x = (ix1-mx0)/abs(mx1-mx0) 
            
            elif ha == "center":
                x = (((ix0-mx0) + (ix1-mx0))/2)/abs(mx1-mx0) 

            else:
                x  = ha      
                ha = "center"
        
        except KeyError:
            ha = 0.5
        
        try:
            va = get_property('plot_title', 'va')
        except KeyError:
            va = "center" 
        
        try:
            ma = get_property('plot_title', 'ma')
        except KeyError:
            ma = "center"

        try:
            margin = get_property('plot_title', 'margin')
        except KeyError:
            pad = 3
        else:
            pad = margin.get_as('b', 'in') / 0.09,
        
        if type(pad) in (list, tuple):
            text = bricks._case.set_title(title, pad=pad[0], fontsize=fontsize, x=x, ha=ha, va=va)
        else:
            text = bricks._case.set_title(title, pad=pad, fontsize=fontsize, x=x, ha=ha, va=va)
        
        if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
            gori.theme._targets['plot_title'] = text
            gori.theme.themeables['plot_title'].apply_figure(gori.figure, gori.theme._targets)
            for ax in gori.axs:
                gori.theme.themeables['plot_title'].apply_ax(ax)
        else:
            gori.figure._themeable['plot_title'] = text
            gori.theme.themeables['plot_title'].apply_figure(gori.figure)
            for ax in gori.axs:
                gori.theme.themeables['plot_title'].apply(ax)
        
    plotnine_version = plotnine.__version__
    
    #save_original_position
    global _basefigure
    global _axes_dict
    position_dict = {} 
    tmp_axes_keys = [key for key in list(_axes_dict.keys()) if type(_axes_dict[key]) == axes.Axes or _axes_dict[key]._type == "Brick"]
    for key in tmp_axes_keys:
        axtmp = _axes_dict[key] 
        position_dict[key] = axtmp.get_position() 

    gcp = copy.deepcopy(ggplot) 
    if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
        figure_subplot_wspace_ori = matplotlib.rcParams["figure.subplot.wspace"]
        figure_subplot_hspace_ori = matplotlib.rcParams["figure.subplot.hspace"]
        figsize_ori = gcp.theme.themeables['figure_size'].properties["value"] 
        if figsize is None:
            figsize = gcp.theme.themeables['figure_size'].properties["value"] 
        matplotlib.rcParams["figure.subplot.wspace"] = figure_subplot_wspace_ori / figsize[0] 
        matplotlib.rcParams["figure.subplot.hspace"] = figure_subplot_hspace_ori / figsize[1] 
        fig, gcp = gcp.draw(return_ggplot=True) 
    
    else:
        fig, gcp = gcp.draw(return_ggplot=True)
        _themeable = fig._themeable
        _basefigure._themeable = _themeable
        figsize_ori = fig.get_size_inches()
        if figsize is None:
            figsize = fig.get_size_inches()  
    
    try:
        strips = gcp.facet.strips
    except:
        strips = []

    ggplot._build()
    axs = ggplot.facet.make_axes(
        _basefigure,
        ggplot.layout.layout,
        ggplot.coordinates) 
    
    ggplot.figure = _basefigure
    ggplot.axs = axs
    
    if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
        ggplot.theme = gcp.theme
        ggplot.theme._targets = gcp.theme._targets

    setmethods = [] 
    for key in dir(gcp.axs[0]):
        if key[0:4] == "set_" and (key != "set_figure" and key != "set_xlabel" and key != "set_ylabel"):
            setmethods.append(key) 
    
    for i in range(len(ggplot.axs)):
        for method in setmethods:
            try:
                getattr(ggplot.axs[i], method)(getattr(gcp.axs[i], "get_" + method[4:])())
            except Exception as e:
                pass 
        if gcp.axs[i]._gridOn == True:
            ggplot.axs[i].grid(c="w") 
        for bar in ["left", "right", "top", "bottom"]:
            ggplot.axs[i].spines[bar].set_lw(gcp.axs[i].spines[bar].get_lw())
            ggplot.axs[i].spines[bar].set_ec(gcp.axs[i].spines[bar].get_ec())
            ggplot.axs[i].spines[bar].set_visible(gcp.axs[i].spines[bar].get_visible())
    
    ggplot._setup_parameters()
    ggplot.facet.strips.generate()  
    for i in range(len(ggplot.facet.strips)):
        if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
            ggplot.facet.strips[i].position = strips[i].draw_info.position
            ggplot.facet.strips[i].draw_info.box_height        = strips[i].draw_info.box_height
            ggplot.facet.strips[i].draw_info.box_width         = strips[i].draw_info.box_width
            ggplot.facet.strips[i].draw_info.strip_text_margin = strips[i].draw_info.strip_text_margin
            ggplot.facet.strips[i].draw_info.strip_align       = strips[i].draw_info.strip_align
            ggplot.facet.strips[i].draw_info.position          = strips[i].draw_info.position
            ggplot.facet.strips[i].draw_info.label             = strips[i].draw_info.label
            ggplot.facet.strips[i].draw_info.rotation          = strips[i].draw_info.rotation
            ggplot.facet.strips[i].draw_info.x                 = strips[i].draw_info.x
            ggplot.facet.strips[i].draw_info.y                 = strips[i].draw_info.y
        else: 
            ggplot.facet.strips[i].info.box_height     = strips[i].info.box_height
            ggplot.facet.strips[i].info.box_width      = strips[i].info.box_width
            ggplot.facet.strips[i].info.box_x          = strips[i].info.box_x
            ggplot.facet.strips[i].info.box_y          = strips[i].info.box_y
            ggplot.facet.strips[i].info.breadth_inches = strips[i].info.breadth_inches
            ggplot.facet.strips[i].info.label          = strips[i].info.label
            ggplot.facet.strips[i].info.location       = strips[i].info.location
            ggplot.facet.strips[i].info.rotation       = strips[i].info.rotation
            ggplot.facet.strips[i].info.x              = strips[i].info.x
            ggplot.facet.strips[i].info.y              = strips[i].info.y 

    #Drawing
    if StrictVersion(plotnine_version) >= StrictVersion("0.12"): 
        from plotnine._mpl.layout_engine import PlotnineLayoutEngine
        from plotnine.themes.themeable import Themeables, themeable
        for i, l in enumerate(ggplot.layers, start=1):
            l.zorder = i + 10
            l.draw(ggplot.layout, ggplot.coordinates)
        ggplot._draw_breaks_and_labels()
        ggplot._draw_watermarks() 
        new = themeable.from_class_name
        ggplot.theme.themeables["figure_size"] = new("figure_size",(1,1))
        ggplot.theme.apply()
    
    elif StrictVersion(plotnine_version) >= StrictVersion("0.9"): 
        ggplot._resize_panels()
        for i, l in enumerate(ggplot.layers, start=1):
            l.zorder = i + 10
            l.draw(ggplot.layout, ggplot.coordinates)
        ggplot._draw_breaks_and_labels()
        ggplot._draw_watermarks() 
        ggplot.theme.apply(ggplot.figure, axs)
    
    elif StrictVersion("0.8") <= StrictVersion(plotnine_version) < StrictVersion("0.9"):
        ggplot._resize_panels()
        ggplot._draw_layers()
        ggplot._draw_breaks_and_labels()
        ggplot._draw_watermarks()
        ggplot._apply_theme()

    else:
        raise ValueError("patchworklib does not support plotnine {}".format(plotnine_version))

    if len(ggplot.axs) == 1: 
        ax = Brick(ax=ggplot.axs[0])
        if "_ggplot_legend" in ax.__dict__:
            ax._ggplot_legend = None #For Google colab... 
        ax.change_plotsize((figsize[0], figsize[1])) 
        
        if StrictVersion(plotnine_version) >= StrictVersion("0.9"):
            xl, yl = draw_labels(ax, ggplot, gcp, figsize) 
            draw_legend(ax, ggplot, gcp, figsize)
            draw_title(ax,  ggplot, gcp, figsize)

        elif StrictVersion("0.8") <= StrictVersion(plotnine_version) < StrictVersion("0.9"):
            draw_labels(ax, ggplot, gcp, figsize) 
            draw_legend(ax, ggplot, gcp, figsize)
            draw_title(ax,  ggplot, gcp, figsize)
        
        else:
            raise ValueError("patchworklib does not support plotnine {}".format(plotnine_version))

        plt.close(fig) 
        del gcp 
        for key in tmp_axes_keys:
            axtmp = _axes_dict[key] 
            axtmp.set_position(position_dict[key])
        
        if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
            ax.set_xlabel(xl) 
            ax.set_ylabel(yl) 
        return_obj = ax 
    
    else:
        width, height = figsize 
        bricks_dict = {}
        row_list = [] 
       
        for ax, axcp in zip(axs, gcp.axs):
            oripos = axcp.get_position()
            ax.set_position([oripos.x0, oripos.y0, oripos.x1-oripos.x0, oripos.y1-oripos.y0])
            brick = Brick(ax=ax) 
            bricks_dict[brick.get_label()] = brick 
        
        bricks = Bricks(bricks_dict=bricks_dict) 
        bricks = expand(bricks, width, height)        
        
        if StrictVersion(plotnine_version) >= StrictVersion("0.9"):
            xl, yl = draw_labels(bricks, ggplot, gcp, figsize) 
            draw_legend(bricks, ggplot, gcp, figsize)
            draw_title(bricks,  ggplot, gcp, figsize)
            pass

        elif StrictVersion("0.8") <= StrictVersion(plotnine_version) < StrictVersion("0.9"):
            draw_labels(bricks, ggplot, gcp, figsize) 
            draw_legend(bricks, ggplot, gcp, figsize)
            draw_title(bricks,  ggplot, gcp, figsize)
        
        else:
            raise ValueError("patchworklib does not support plotnine {}".format(plotnine_version))

        plt.close(fig) 
        del gcp 
        for key in tmp_axes_keys:
            ax = _axes_dict[key] 
            ax.set_position(position_dict[key])    

        x0, x1, y0, y1 = bricks.get_outer_corner() 
        bricks._originalsize = (abs(x1-x0), abs(y0-y1))
        bricks.set_originalpositions() 
        if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
            bricks.case.set_xlabel(xl) 
            bricks.case.set_ylabel(yl) 
        return_obj =  bricks
    
    if StrictVersion(plotnine_version) >= StrictVersion("0.12"):
        matplotlib.rcParams["figure.subplot.wspace"] = figure_subplot_wspace_ori 
        matplotlib.rcParams["figure.subplot.hspace"] = figure_subplot_hspace_ori  
        return_obj.savefig(_ggplot=True)
        
    return return_obj

def overwrite_axisgrid():
    """

    Overwrite `__init__` functions in seaborn.axisgrid.FacetGrid, 
    seaborn.axisgrid.PairGrid and seaborn.axisgrid.JointGrid.
    The function changes the figure object given in the `__init__` functions of the 
    axisgrid class objects, which is used for drawing plots, to `_basefigure 
    in the patchworklib. If you want to import plots generated baseon 
    seabron.axisgrid.xxGrid objects as patchworklib.Brick(s) object by using 
    `load_seaborngrid` function, you should execute the function in advance.

    Returns
    -------
    None.
    
    """ 

    #sns.pairplot = mg.pairplot
    sns.axisgrid.Grid._figure          = _basefigure
    sns.axisgrid.Grid.add_legend       = mg.add_legend
    sns.axisgrid.FacetGrid.__init__    = mg.__init_for_facetgrid__
    sns.axisgrid.FacetGrid.despine     = mg.despine 
    sns.axisgrid.PairGrid.__init__     = mg.__init_for_pairgrid__
    sns.axisgrid.JointGrid.__init__    = mg.__init_for_jointgrid__
    sns.matrix.ClusterGrid.__init__    = mg.__init_for_clustergrid__ 
    sns.matrix.ClusterGrid.__setattr__ = mg.__setattr_for_clustergrid__ 
    sns.matrix.ClusterGrid.plot        = mg.__plot_for_clustergrid__

def load_seabornobj(g, label=None, labels=None, figsize=(3,3)): 
    """
    Load seaborn plot generated based on seaborn._core.plot.Plotter class. 
    The method is prototype version. By using this function, plots generated 
    using next generation seaborn interface can be handled as 
    patchworklib.Brick(s) class object.
    
    Parameters
    ----------
    g : seaborn.axisgrid.FacetGrid, seaborn.axisgrid.PairGrid or seaborn.axisgrid.JointGrid object
        A return value of the seaborn figure-level plotting functions such as relplot, 
        distplot, catplot, jointplot and pairplot.
    label : str, 
        Unique identifier for the patchworklib.Bricks class object to be returned. If you 
        want to adjust a layout composed of multiple bricks object using the label 
        indexing method, providing the unique label name for returned object should be 
        encouraged.
    figsize : tuple (float, float), 
        Figure size. If it is not specified, convert Bricks object would keep the original 
        figure size of the given seaborn plot. 
    
    Returns
    -------
    None.
 
    """
    global _basefigure
    if len(g._subplots._subplot_list) == 1:
        aplot  = g._subplots._subplot_list[0] 
        bricks = Brick(ax=aplot["ax"], label=label)
        bricks.change_plotsize(figsize)  

    else:
        plots = [] 
        for aplot in g._subplots._subplot_list:
            plots.append(aplot["ax"]) 
        
        bricks_dict = {} 
        for i, ax in enumerate(plots):
            if labels is None:
                brick = Brick(ax=ax)
            else:
                brick = Brick(ax=ax, label=labels[i]) 
            bricks_dict[brick.get_label()] = brick
        
        if label is None: 
            bricks = Bricks(bricks_dict) 
        else:
            bricks = Bricks(bricks_dict, label=label)
     
    merged_contents: dict[
        tuple[str | None, str | int], tuple[list[Artist], list[str]],
    ] = {}
    for key, artists, labels in g._legend_contents:
        if key in merged_contents:
            existing_artists = merged_contents[key][0]
            for i, artist in enumerate(existing_artists):
                if isinstance(artist, tuple):
                    artist += artist[i],
                else:
                    existing_artists[i] = artist, artists[i]
        else:
            merged_contents[key] = artists.copy(), labels
    
    i = 0 
    _basefigure.legends = [] 
    for (name, _), (handles, labels) in merged_contents.items():
        if i == 0:
            baselegend = bricks.case.legend(
                handles = handles,
                labels  = labels,
                title=name,                  
                loc="center left",
                bbox_to_anchor=(1. + (0.3 / figsize[0]), 0.5)
            )
        else:
            legend = matplotlib.legend.Legend(
                _basefigure,
                handles,
                labels,
                title=name,                  
                loc="center left",
                bbox_to_anchor=(1. + (0.1 / figsize[0]), 0.5)
            )
            baselegend._legend_box._children.extend(legend._legend_box._children)
        i += 1
    
    
    if len(g._subplots._subplot_list) > 1:
        outers = bricks.get_inner_corner() 
        expand(bricks, figsize[0]/abs(outers[0]-outers[1]), figsize[1]/abs(outers[3]-outers[2])) 
        x0, x1, y0, y1 = bricks.get_outer_corner() 
        bricks._originalsize = (abs(x1-x0), abs(y0-y1))
        bricks.set_originalpositions()
    return bricks 

def load_seaborngrid(g, label=None, labels=None, figsize=None):
    """

    Load seaborn plot generated based on seaborn.axisgrid.xxGrid class.   
    In generally, seaborn plots generated by figure-level fucntion cannot be 
    handles as subplot(s) with other matplotlib plots, However, by processing 
    these seaborn plots via the function, you can handle them as 
    patchworklib.Brick(s) class objects. 
    
    Notes
    -----
    You should execute `overwrite_axisgrid` function before using this function.

    Parameters
    ----------
    g : seaborn.axisgrid.FacetGrid, seaborn.axisgrid.PairGrid or seaborn.axisgrid.JointGrid object
        A return value of the seaborn figure-level plotting functions such as relplot, 
        distplot, catplot, jointplot and pairplot.
    label : str, 
        Unique identifier for the patchworklib.Bricks class object to be returned. If you 
        want to adjust a layout composed of multiple bricks object using the label 
        indexing method, providing the unique label name for returned object should be 
        encouraged.
    figsize : tuple (float, float), 
        Figure size. If it is not specified, convert Bricks object would keep the original 
        figure size of the given seaborn plot. 
    
    Returns
    -------
    None.
    
    """
    bricks_dict = {} 
    if type(g) == sns.axisgrid.JointGrid:
        axes = [g.ax_joint, g.ax_marg_x, g.ax_marg_y] 
    
    elif type(g) == sns.matrix.ClusterGrid:
        axes = [] 
        axes.append(g.ax_heatmap) 
        if g.ax_row_colors is None:
            pass 
        else:
            axes.append(g.ax_row_colors)
        
        if g.ax_col_colors is None:
            pass 
        else:
            axes.append(g.ax_col_colors)
        
        if g.dendrogram_row is None:
            pass 
        else:
            axes.append(g.ax_row_dendrogram)
        
        if g.dendrogram_col is None:
            pass 
        else:
            axes.append(g.ax_col_dendrogram)

        if g.ax_cbar is None:
            pass 
        else:
            positions = [ax.get_position() for ax in axes] 
            min_x0   = min([pos.x0 for pos in positions]) 
            max_x1   = max([pos.x1 for pos in positions]) 
            min_y0   = min([pos.y0 for pos in positions]) 
            max_y1   = max([pos.y1 for pos in positions]) 
            cbar_pos = g.ax_cbar.get_position() 
            
            if cbar_pos.x1 <= min_x0:
                new_cx0 = min_x0 - abs(cbar_pos.x1 - min_x0)/g._figsize[0] - abs(cbar_pos.x0-cbar_pos.x1) 
                new_cx1 = min_x0 - abs(cbar_pos.x1 - min_x0)/g._figsize[0]
            
            elif cbar_pos.x0 > max_x1:
                new_cx0 = max_x1 + abs(cbar_pos.x0 - max_x1)/g._figsize[0] 
                new_cx1 = max_x1 + abs(cbar_pos.x0 - max_x1)/g._figsize[0] + abs(cbar_pos.x0-cbar_pos.x1) 
            else:
                new_cx0 = cbar_pos.x0 
                new_cx1 = cbar_pos.x1

            if cbar_pos.y1 <= min_y0:
                new_cy0 = min_y0 - abs(cbar_pos.y1 - min_y0)/g._figsize[1] - abs(cbar_pos.y0-cbar_pos.y1) 
                new_cy1 = min_y0 - abs(cbar_pos.y1 - min_y0)/g._figsize[1]
            
            elif cbar_pos.y0 > max_y1:
                new_cy0 = max_y1 + abs(cbar_pos.y0 - max_y1)/g._figsize[1] 
                new_cy1 = max_y1 + abs(cbar_pos.y0 - max_y1)/g._figsize[1] + abs(cbar_pos.y0-cbar_pos.y1) 
            else:
                new_cy0 = cbar_pos.y0 
                new_cy1 = cbar_pos.y1
            
            g.ax_cbar.set_position([new_cx0, new_cy0, new_cx1-new_cx0, new_cy1-new_cy0])
            axes.append(g.ax_cbar)

    else:
        axes = g.axes.tolist()
        if type(axes[0]) == list:
            axes = sum(axes, [])
    
    if "diag_axes" in g.__dict__:
        if g.__dict__["diag_axes"] is None:
            pass 
        else:
            if type(g.diag_axes) == list: 
                diag_axes = g.diag_axes
            else:
                diag_axes = g.diag_axes.tolist()
            axes.extend(diag_axes) 

    for i, ax in enumerate(axes):
        if labels is None:
            brick = Brick(ax=ax)
        else:
            brick = Brick(ax=ax, label=labels[i]) 
        bricks_dict[brick.get_label()] = brick
    
    if label is None: 
        bricks = Bricks(bricks_dict) 
    else:
        bricks = Bricks(bricks_dict, label=label)

    if "_figlegend" in g.__dict__:
        legend_set = g._figlegend
        if "loc" in legend_set[2] and legend_set[2]["loc"] == "center right":
            legend_set[2]["loc"] = "center left"
        legend = bricks.case.legend(legend_set[0], legend_set[1], **legend_set[2], bbox_to_anchor=(1. + (0.1 / g._figsize[0]), 0.5))
        legend.set_title(legend_set[3], prop={"size": legend_set[4]})
        bricks._seaborn_legend = (legend_set[2], (1. + (0.1 / g._figsize[0]), 0.5)) 
    
    outers = bricks.get_inner_corner() 
    if figsize is None:
        expand(bricks, g._figsize[0]/abs(outers[0]-outers[1]), g._figsize[1]/abs(outers[3]-outers[2])) 
    else:
        expand(bricks, figsize[0]/abs(outers[0]-outers[1]), figsize[1]/abs(outers[3]-outers[2])) 
    
    x0, x1, y0, y1 = bricks.get_outer_corner() 
    bricks._originalsize = (abs(x1-x0), abs(y0-y1))
    bricks.set_originalpositions() 

    return bricks 

def clear():
    """

    Claer all patchworklib.Brick(s) objects generated so far.

    Returns
    -------
    None.

    """ 
    global _axes_dict
    global _removed_axes
    global _basefigure
    for label, ax in _removed_axes.items():
        ax.figure = _basefigure
        _basefigure.add_axes(ax)
    _axes_dict = {}
    for ax in _basefigure.axes:
        ax.remove() 
        del ax 
    _removed_axes = {} 
    pBrick._labelset = set([]) 

def inset(brick1, brick2, loc="upper right", wratio=0.4, hratio=0.4, vmargin=0.1, hmargin=0.1, alpha=0.0): 
    """

    Arrange a brick2 object in the brick1 object.
    
    Parameters
    ----------
    brick1 : patchworklib.Brick or patchworklib.Bricks class object 
        Brick(s) class object to be joined with `brick2` object. The location of this 
        object is used as the base position for determining the `brick2` placement. 
    brick2 : patchworklib.Brick or patchworklib.Bricks class object
        Brick(s) class object to be placed in specified in `brick1` object. 
    position : str, ("upper right", "lower rigtht", "upper left", lower left") 
        Position of `brick2` object in `brick1` object. 
    wratio : float, default: 0.1
        Ratio of the `brick2` width to `brick1` object. 
    hratio : float, default: 0.1
        Ratio of the `brick2` height to `brick1` object. 
    vmargin : float, default: 0.1
        Margin from the bottom/top.
    hmargin : float, default: 0.1
        Margin from the right/left.
    alpha : flaot, default: 0.0
        Alpha of background of `brick2` object 
    
    Returns
    -------
    patchworlib.Bricks class object
    """
    
    if vmargin is None:
        vmargin = param["margin"]
            
    if hmargin is None:
        hmargin = param["margin"] 

    inner1 = brick1.get_inner_corner() 
    iw1 = abs(inner1[1]-inner1[0])
    ih1 = abs(inner1[3]-inner1[2])

    inner2 = brick2.get_inner_corner() 
    iw2 = abs(inner2[1]-inner2[0])
    ih2 = abs(inner2[3]-inner2[2])
    
    expand(brick2, iw1*wratio/iw2, ih1*hratio/ih2)  
    outer2 = brick2.get_outer_corner() 

    if loc == "upper right":
        topright = (inner1[1]-hmargin, inner1[3]-vmargin) 
        shifth = -1 * (outer2[1] - topright[0])
        shiftv = -1 * (outer2[3] - topright[1])
    elif loc == "lower right":
        bottomright = (inner1[1]-hmargin, inner1[2]+vmargin) 
        shifth = -1 * (outer2[1] - bottomright[0])
        shiftv = -1 * (outer2[2] - bottomright[1])
    elif loc == "upper left":
        topleft = (inner1[0]+hmargin, inner1[3]-vmargin) 
        shifth = -1 * (outer2[0] - topleft[0])
        shiftv = -1 * (outer2[3] - topleft[1])
    elif loc == "lower left":
        bottomleft = (inner1[0]+hmargin, inner1[2]+vmargin) 
        shifth = -1 * (outer2[0] - bottomleft[0])
        shiftv = -1 * (outer2[2] - bottomleft[1])
    else:
        pass 
    
    zorders = [] 
    for key in brick1.bricks_dict:
        zorders.append(brick1.bricks_dict[key].get_zorder()) 
    
    if brick2._type == "Bricks":
        brick2._case.set_zorder(max(zorders)+1) 
        for art in brick2._case.artists: 
            art.set_zorder(max(zorders)+1)

    for key in brick2.bricks_dict:
        ax  = brick2.bricks_dict[key] 
        ax.set_zorder(max(zorders)+2) 
        for art in ax.artists:
            art.set_zorder(max(zorders)+2)

        pos = ax.get_position()
        ax.set_position([pos.x0 + shifth, pos.y0 + shiftv, pos.x1-pos.x0, pos.y1-pos.y0])    
        _reset_ggplot_legend(ax)
    
    if alpha > 0.0:
        brick2._case.set_zorder(max(zorders)+1) 
        brick2._case.patch.set_alpha(alpha) 

    labels_all = list(brick1._labels) + list(brick2._labels) 
    for caselabel in brick2._case_labels:
        caselabel = caselabel[5:] 
        _reset_ggplot_legend(_axes_dict[caselabel])

    bricks_dict = {}
    for key in brick1.bricks_dict:
        bricks_dict[key] = brick1.bricks_dict[key] 
    
    for key in brick2.bricks_dict:
        bricks_dict[key] = brick2.bricks_dict[key]
    
    new_bricks = Bricks(bricks_dict)
    new_bricks._brick1  = _axes_dict[brick1._label]
    new_bricks._brick2  = _axes_dict[brick2._label]
    new_bricks._command = "inset"
    new_bricks._target  = None
    
    for label in labels_all:
        if "_case" in _axes_dict[label].__dict__:
            _axes_dict[label].case

    new_bricks._case_labels = new_bricks._case_labels + brick1._case_labels + brick2._case_labels
    for label in labels_all:
        new_bricks._labels.add(label)
    
    return new_bricks

def hstack(brick1, brick2, target=None, margin=None, direction="r", adjust_height=True, adjust_width=True, va="top"):
    """
    Align two patchworlib.Brick(s) objects horizontally.
    When joining two Brick(s) objects by "|" operator, this function is 
    called internally. Thus, `x|y` means `hstack(x,y)`.

    Parameters
    ----------
    brick1 : patchworklib.Brick or patchworklib.Bricks class object 
        Brick(s) class object to be joined with `brick2` object. The location of this 
        object is used as the base position for determining the `brick2` placement. 
    brick2 : patchworklib.Brick or patchworklib.Bricks class object
        Brick(s) class object to be placed on the side specified by `direction` 
        (by default, on the right side) of the `brick1` object.
    target : str, default: None, 
        Unique label name of the Brick or Brick(s) object that is a part of the `brick1` 
        object. If you want to place `brick2` object next to the specific Brick(s) object 
        in `brick1` object, please provide the `label` value of the Brick(s) object.
    margin : float or str ("none"), default: None
        Margin size between the two given Brick(s) objects. If None, the 
        `pw.param["margin"]` value would be used as the margin size. If the value is "none", 
        two Brick(s) objects will be joined with no margin (meaning that the axes spines 
        will be joined so that they are fully glued together). 
    direction : str ("r" or "l"), default: "r"
        Side on which `brick2` is placed with respect to `brick1`. 
        "r" means right. "l" means left.
    adjust_height : bool, default: True 
        If True, the height of `brick2` object is adjusted so that it will be equal to the 
        height of `brick1` object.
    adjust_width : bool, default: True
        If True, the width of `brick2` will be adjusted according to the aspect of `brick2`
        after stacking. If False, `brick2` will keep its original width after stacking. 
        If ajust_height is Fasle, the value will also be False. 
    va : str ("top" or "bottom"), default: False
       If adjsut_height is False, the value will be effective. If the value is "top", 
       `brick2` object will be aligned to the left/right top of `brick1` object. If 
       "bottom", `brick2` object will be aligned to the left/right bottom of `brick1` 
       object.

    Returns
    -------
    patchworklib.Bricks class object

    """

    global param 
    global _axes_dict 
    
    labels_all = list(brick1._labels) + list(brick2._labels)
    for label in labels_all:
        if "_case" in _axes_dict[label].__dict__:
            _axes_dict[label].case

    ax_adjust = None
    if margin is None:
        margin = param["margin"] 
    
    if margin == "none":
        margin = None

    if brick1._type == "Bricks":
        for label in brick1._originalpositions:
            x0, x1, y0, y1 = brick1._originalpositions[label]
            _axes_dict[label].set_position([x0, y0, abs(x0-x1), abs(y0-y1)]) 
        
        for case_label in brick1._case_labels:
            _axes_dict[":".join(case_label.split(":")[1:])].case 

    elif brick1._type == "Brick":
        brick1.set_position([0, 0, brick1._originalsize[0], brick1._originalsize[1]]) 
        brick1._parent = None  
        target = None
        labels = None
    
    if brick2._type == "Bricks":
        for label in brick2._originalpositions:
            x0, x1, y0, y1 = brick2._originalpositions[label]
            _axes_dict[label].set_position([x0, y0, abs(x0-x1), abs(y0-y1)]) 
        
        for case_label in brick2._case_labels:
            _axes_dict[":".join(case_label.split(":")[1:])].case 

    elif brick2._type == "Brick":
        brick2._parent = None  
        brick2.set_position([0, 0, brick2._originalsize[0], brick2._originalsize[1]]) 

    if target is not None:
        parent = brick1
        brick1_bricks_dict = brick1.bricks_dict
        if type(target) is str:
            if target in brick1.bricks_dict:
                brick1 = brick1.bricks_dict[target]
                brick1._parent = None
                labels = None
            else:
                brick1 = _axes_dict[target] 
                brick1._parent = None
                labels = [key for key in brick1.bricks_dict] 
            
        elif type(target) is tuple:
            if type(target[0]) is str:
                labels = target
            else:
                labels = [t._label for t in target]
        
        else:
            brick1 = target
            brick1._parent = None
            if brick1._type == "Brick":
                labels = None
            else:
                labels = list(brick1.bricks_dict.keys())
        
    else:
        brick1_bricks_dict = brick1.bricks_dict
        labels = None 
    
    if va not in ("top", "bottom", "t", "b"):
        raise ValueError("'va' value should be 't', 'top', 'b' or 'bottom'.")
    
    if labels is None:
        brick1_ocorners = brick1.get_outer_corner() 
    else:
        brick1_ocorners = brick1.get_middle_corner(labels)
    
    brick2_ocorners = brick2.get_outer_corner() 
    brick1_icorners = brick1.get_inner_corner(labels)  
    brick2_icorners = brick2.get_inner_corner() 
    vratio = abs(brick1_icorners[3] - brick1_icorners[2]) / abs(brick2_icorners[3] - brick2_icorners[2])  
    if vratio < 0.8 and target is None: 
        expand(brick1, 1/vratio, 1/vratio) 
        brick1_ocorners = brick1.get_outer_corner() 
        brick2_ocorners = brick2.get_outer_corner() 
        brick1_icorners = brick1.get_inner_corner()  
        brick2_icorners = brick2.get_inner_corner() 
        vratio = abs(brick1_icorners[3] - brick1_icorners[2]) / abs(brick2_icorners[3] - brick2_icorners[2])  
    
    if adjust_height == True and adjust_width == True:
        expand(brick2, vratio, vratio) 
    elif adjust_height == True:  
        expand(brick2, 1, vratio)
    else:
        pass 

    if target is not None: 
        parent_icorners = parent.get_inner_corner()
        brick2_icorners = brick2.get_inner_corner() 
        if direction == "r":
            hlength = parent_icorners[1] - ((margin if margin is not None else 0) + brick1_ocorners[1] + abs(brick2_ocorners[0]-brick2_icorners[0]))
        else:
            hlength = -1 * (parent_icorners[0] - (brick1_ocorners[0] - (margin if margin is not None else 0) - abs(brick2_ocorners[0] - brick2_icorners[0]))) 

        if hlength > 0:
            keys = [] 
            for key in parent.bricks_dict:
                ax = parent.bricks_dict[key]
                ic = ax.get_inner_corner()
                if direction == "r":
                    dist = ic[0] - brick1_ocorners[1]
                    if dist > 0:
                        keys.append((dist,key))
                else:
                    dist = ic[1] - brick1_ocorners[0]
                    if dist < 0:
                        keys.append((abs(dist),key))

            if len(keys) > 0:
                keys.sort()
                if direction == "r":
                    hratio  = (parent.bricks_dict[keys[-1][1]].get_inner_corner()[1] -  parent.bricks_dict[keys[0][1]].get_inner_corner()[0]) / abs(brick2_icorners[1] - brick2_icorners[0])  
                else:
                    hratio  = abs(parent.bricks_dict[keys[-1][1]].get_inner_corner()[0] -  parent.bricks_dict[keys[0][1]].get_inner_corner()[1]) / abs(brick2_icorners[1] - brick2_icorners[0])  
                
                if direction == "r":
                    ax_adjust = parent.bricks_dict[keys[0][1]]
                else:
                    ax_adjust = parent.bricks_dict[keys[-1][1]]
                 
                for key in brick2.bricks_dict:
                    ax  = brick2.bricks_dict[key] 
                    pos = ax.get_position()
                    ax.set_position([pos.x0 * hratio, pos.y0, abs(pos.x1-pos.x0) * hratio, abs(pos.y1-pos.y0)]) 
                    _reset_ggplot_legend(ax)     

            for caselabel in brick2._case_labels:
                caselabel = caselabel[5:] 
                _reset_ggplot_legend(_axes_dict[caselabel])

    brick1_ocorners = brick1.get_outer_corner() 
    brick2_ocorners = brick2.get_outer_corner() 
    brick1_icorners = brick1.get_inner_corner()  
    brick2_icorners = brick2.get_inner_corner() 
    for key in brick2.bricks_dict:
        ax  = brick2.bricks_dict[key] 
        pos = ax.get_position()
        if adjust_height == False and va in ("t", "top"):
            base = brick1_icorners[3] - (brick2_icorners[3]-brick2_icorners[2]) 
        else:
            base = brick1_icorners[2] 

        if ax_adjust is None:
            if margin is not None:
                if direction == "r":
                    ax.set_position([margin + brick1_ocorners[1] + abs(brick2_ocorners[0]-brick2_icorners[0]) + pos.x0 - brick2_icorners[0], pos.y0 - brick2_icorners[2] + base, pos.x1-pos.x0, pos.y1-pos.y0])
                elif direction == "l":
                    ax.set_position([abs(brick2_ocorners[0]-brick2_icorners[0]) + pos.x0 - brick2_icorners[0] + brick1_ocorners[0] - margin - (brick2_ocorners[1]-brick2_ocorners[0]), pos.y0 - brick2_icorners[2] + base, pos.x1-pos.x0, pos.y1-pos.y0])
            else:
                if direction == "r":
                    ax.set_position([brick1_icorners[1] + pos.x0 - brick2_icorners[0], pos.y0 - brick2_icorners[2] + base, pos.x1-pos.x0, pos.y1-pos.y0])
                elif direction == "l":
                    ax.set_position([pos.x0 - brick2_icorners[0] + brick1_icorners[0] - (brick2_icorners[1]-brick2_icorners[0]), pos.y0 - brick2_icorners[2] + base, pos.x1-pos.x0, pos.y1-pos.y0])
        else:
            if direction == "r":
                ax.set_position([ax_adjust.get_inner_corner()[0] + pos.x0 - brick2_icorners[0], pos.y0 - brick2_icorners[2] + base, pos.x1-pos.x0, pos.y1-pos.y0])
            elif direction == "l":
                ax.set_position([ax_adjust.get_inner_corner()[0] + pos.x0 - brick2_icorners[0], pos.y0 - brick2_icorners[2] + base, pos.x1-pos.x0, pos.y1-pos.y0])
            
        pos = ax.get_position()
        _reset_ggplot_legend(ax)
    
    for caselabel in brick2._case_labels:
        caselabel = caselabel[5:] 
        _reset_ggplot_legend(_axes_dict[caselabel])

    bricks_dict = {}
    for key in brick1_bricks_dict:
        if "outline:" == key[:len("outline:")]:
            pass 
        else:
            bricks_dict[key] = brick1_bricks_dict[key] 
    
    for key in brick2.bricks_dict:
        if "outline:" == key[:len("outline:")]:
            pass 
        else:
            bricks_dict[key] = brick2.bricks_dict[key]
    
    new_bricks = Bricks(bricks_dict)
    new_bricks._brick1  = _axes_dict[brick1._label]
    new_bricks._brick2  = _axes_dict[brick2._label]
    new_bricks._command = "hstack"
    if target is None:
        new_bricks._target = None
    else:
        new_bricks._target = _axes_dict[brick1._label]
    
    for label in labels_all:
        if "_case" in _axes_dict[label].__dict__:
            _axes_dict[label].case
    
    if target is not None: 
        new_bricks._case_labels = new_bricks._case_labels + parent._case_labels + brick2._case_labels
    else:
        new_bricks._case_labels = new_bricks._case_labels + brick1._case_labels + brick2._case_labels
    
    for label in labels_all:
        new_bricks._labels.add(label) 
    return new_bricks

def vstack(brick1, brick2, target=None, margin=None, direction="t", adjust_height=True, adjust_width=True, ha="l"):
    """
    Align two patchworlib.Brick(s) objects vertically.
    When joining two Brick(s) objects by "/" operator, this function is called 
    internally. `brick2 / brick1` means `vstack(brick1, brick2)`

    Parameters
    ----------
    brick1 : patchworklib.Brick or patchworklib.Bricks class object 
        Brick(s) class object to be joined with `brick2` object. The location of this 
        object is used as the base position for determining the `brick2` placement.
    brick2 : patchworklib.Brick or patchworklib.Bricks class object
        Brick(s) class object to be placed on the side specified by `direction` 
        (by default, on the top side) of the `brick1` object.
    target : str, default: None 
        Unique label name of the Brick or Brick(s) object that is a part of the `brick1` 
        object. If you want to place `brick2` object next to the specific Brick(s) object 
        in `brick1` object, please provide the `label` value of the Brick(s) object.
    margin : flaot, default: `pw.param["margin"]` 
        Margin size between the two given Brick(s) objects. If None, the 
        `pw.param["margin"]` value would be used as the margin size. If the value is 
        "none", two Brick(s) objects will be joined with no margin (meaning that the axes 
        spines will be joined so that they are fully glued together). 
    direction : str ("t" or "b"), default: "t"
        Side on which `brick2` is placed with respect to `brick1`. 
        "t" means top, "b" means bottom.
    adjust_height : bool, default: True 
        If True, the height of `brick2` will be adjusted according to the aspect of `brick2`
        after stacking. If False, `brick2` will keep its original height after stacking. 
        If ajust_width is Fasle, the value will also be False. 
    adjust_width : bool, default: True
        If True, the width of `brick2` object is adjusted so that it will be equal to the 
        width of `brick1` object.
    ha : str ("top" or "bottom"), default: False
       If adjsut_width is False, the value will be effective. If the value is "", 
       `brick2` object will be aligned to the left/right top of `brick1` object. 

    Returns
    -------
    patchworlib.Bricks class object

    """
 
    global param
    global _axes_dict

    labels_all = list(brick1._labels) + list(brick2._labels)
    for label in labels_all:
        if "_case" in _axes_dict[label].__dict__:
            _axes_dict[label].case

    if margin is None:
        margin = param["margin"] 
    
    if margin == "none":
        margin = None

    ax_adjust = None
    if brick1._type == "Bricks":
        for label in brick1._originalpositions:
            x0, x1, y0, y1 = brick1._originalpositions[label]
            _axes_dict[label].set_position([x0, y0, abs(x0-x1), abs(y0-y1)]) 
        
        for case_label in brick1._case_labels:
            _axes_dict[":".join(case_label.split(":")[1:])].case 

    elif brick1._type == "Brick":
        brick1.set_position([0, 0, brick1._originalsize[0], brick1._originalsize[1]]) 
        brick1._parent = None
        labels = None
    
    if brick2._type == "Bricks":
        for label in brick2._originalpositions:
            x0, x1, y0, y1 = brick2._originalpositions[label]
            _axes_dict[label].set_position([x0, y0, abs(x0-x1), abs(y0-y1)]) 
        
        for case_label in brick2._case_labels:
            _axes_dict[":".join(case_label.split(":")[1:])].case 

    elif brick2._type == "Brick":
        brick2._parent = None
        brick2.set_position([0, 0, brick2._originalsize[0], brick2._originalsize[1]]) 
    
    if target is not None:
        parent = brick1
        brick1_bricks_dict = brick1.bricks_dict
        
        if type(target) in (Bricks, Brick, cBrick): 
            target = target.get_label() 

        if type(target) is str:
            if target in brick1.bricks_dict:
                brick1 = brick1.bricks_dict[target]
                brick1._parent = None
                labels = None
            else:
                brick1 = _axes_dict[target] 
                brick1._parent = None
                labels = [key for key in brick1.bricks_dict]
         
        elif type(target) is tuple:
            if type(target[0]) is str:
                labels = target
            else:
                labels = [t._label for t in target]
        else:
            brick1 = target
            brick1._parent = None
            if brick1._type == "Brick":
                labels = None
            else:
                labels = list(brick1.bricks_dict.keys())
    else:
        brick1_bricks_dict = brick1.bricks_dict
        labels = None
    
    if ha not in ("l", "left", "r", "right"):
        raise ValueError("'ha' value should be 'l', 'left', 'r' or 'right'.")

    if labels is None:
        brick1_ocorners = brick1.get_outer_corner() 
    else:
        brick1_ocorners = brick1.get_middle_corner(labels)
    
    brick2_ocorners = brick2.get_outer_corner() 
    brick1_icorners = brick1.get_inner_corner(labels)  
    brick2_icorners = brick2.get_inner_corner() 
    hratio = abs(brick1_icorners[1] - brick1_icorners[0]) / abs(brick2_icorners[1] - brick2_icorners[0])  
    if hratio < 1.0 and target is None: 
        expand(brick1, 1/hratio, 1/hratio) 
        brick1_ocorners = brick1.get_outer_corner() 
        brick2_ocorners = brick2.get_outer_corner() 
        brick1_icorners = brick1.get_inner_corner()  
        brick2_icorners = brick2.get_inner_corner() 
        hratio = abs(brick1_icorners[1] - brick1_icorners[0]) / abs(brick2_icorners[1] - brick2_icorners[0])  
    
    if adjust_width == True and adjust_height == True:
        expand(brick2, hratio, hratio) 
    elif adjust_width == True:
        expand(brick2, hratio, 1)
    else:
        pass 

    if target is not None: 
        parent_icorners = parent.get_inner_corner()
        brick2_icorners = brick2.get_inner_corner() 
        if direction == "t":
            vlength = parent_icorners[3] - ((margin if margin is not None else 0) + brick1_ocorners[3] + abs(brick2_ocorners[2] - brick2_icorners[2])) 
        else:
            vlength = -1 * (parent_icorners[2] - (brick1_icorners[2] - (margin if margin is not None else 0) - abs(brick2_ocorners[2] - brick2_icorners[2]))) 
        
        if vlength > 0:
            keys = [] 
            for key in parent.bricks_dict:
                ax = parent.bricks_dict[key]
                ic = ax.get_inner_corner()
                if direction == "t":
                    dist = ic[2] - brick1_ocorners[3]
                    if dist > 0:
                        keys.append((dist,key))
                else:
                    dist = ic[3] - brick1_ocorners[2]
                    if dist < 0:
                        keys.append((abs(dist),key))
            
            if len(keys) > 0:
                keys.sort()
                if direction == "t":
                    vratio  = (parent.bricks_dict[keys[-1][1]].get_inner_corner()[3] -  parent.bricks_dict[keys[0][1]].get_inner_corner()[2]) / abs(brick2_icorners[3] - brick2_icorners[2])  
                else:
                    vratio  = abs(parent.bricks_dict[keys[-1][1]].get_inner_corner()[2] -  parent.bricks_dict[keys[0][1]].get_inner_corner()[3]) / abs(brick2_icorners[3] - brick2_icorners[2])  
                
                if direction == "t":
                    ax_adjust = parent.bricks_dict[keys[0][1]]
                else:
                    ax_adjust = parent.bricks_dict[keys[-1][1]]

                for key in brick2.bricks_dict:
                    ax  = brick2.bricks_dict[key] 
                    pos = ax.get_position()
                    ax.set_position([pos.x0, pos.y0 * vratio,  abs(pos.x1-pos.x0), abs(pos.y1-pos.y0) * vratio]) 
                    _reset_ggplot_legend(ax)     
            
            for caselabel in brick2._case_labels:
                caselabel = caselabel[5:] 
                _reset_ggplot_legend(_axes_dict[caselabel])

       
    brick1_ocorners = brick1.get_outer_corner() 
    brick2_ocorners = brick2.get_outer_corner() 
    brick1_icorners = brick1.get_inner_corner()  
    brick2_icorners = brick2.get_inner_corner() 
    for key in brick2.bricks_dict:
        ax  = brick2.bricks_dict[key] 
        pos = ax.get_position()
        if adjust_width == False and ha in ("r", "right"):
            base = brick1_icorners[1] - (brick2_icorners[1]-brick2_icorners[0]) 
        else:
            base = brick1_icorners[0] 

        if ax_adjust is None:
            if margin is not None:
                if direction == "t":
                    ax.set_position([pos.x0 - brick2_icorners[0] + base, margin + pos.y0 - brick2_icorners[2] + brick1_ocorners[3] + abs(brick2_ocorners[2] - brick2_icorners[2]) , pos.x1-pos.x0, pos.y1-pos.y0])
                elif direction == "b":
                    ax.set_position([pos.x0 - brick2_icorners[0] + base, pos.y0 - brick2_icorners[2] + abs(brick2_ocorners[2] - brick2_icorners[2]) - margin + brick1_ocorners[2] - (brick2_ocorners[3]-brick2_ocorners[2]), pos.x1-pos.x0, pos.y1-pos.y0])
            else:
                if direction == "t":
                    ax.set_position([pos.x0 - brick2_icorners[0] + base, pos.y0 - brick2_icorners[2] + brick1_icorners[3], pos.x1-pos.x0, pos.y1-pos.y0])
                elif direction == "b":
                    ax.set_position([pos.x0 - brick2_icorners[0] + base, pos.y0 - brick2_icorners[2] + brick1_icorners[2] - (brick2_icorners[3]-brick2_icorners[2]), pos.x1-pos.x0, pos.y1-pos.y0])
        else:
            if direction == "t":
                ax.set_position([pos.x0 - brick2_icorners[0] + base, pos.y0 - brick2_icorners[2] + ax_adjust.get_inner_corner()[2], pos.x1-pos.x0, pos.y1-pos.y0])
            elif direction == "b":
                ax.set_position([pos.x0 - brick2_icorners[0] + base, pos.y0 - brick2_icorners[2] + ax_adjust.get_inner_corner()[2], pos.x1-pos.x0, pos.y1-pos.y0])
            
        pos = ax.get_position() 
        _reset_ggplot_legend(ax)

    for caselabel in brick2._case_labels:
        caselabel = caselabel[5:] 
        _reset_ggplot_legend(_axes_dict[caselabel])

    bricks_dict = {}
    for key in brick1_bricks_dict:
        if "outline:" == key[:len("outline:")]:
            pass 
        else:
            bricks_dict[key] = brick1_bricks_dict[key] 
    
    for key in brick2.bricks_dict:
        if "outline:" == key[:len("outline:")]:
            pass 
        else:
            bricks_dict[key] = brick2.bricks_dict[key]
    
    new_bricks = Bricks(bricks_dict) 
    new_bricks._brick1  = _axes_dict[brick1._label]
    new_bricks._brick2  = _axes_dict[brick2._label]
    new_bricks._command = "vstack"
    if target is None:
        new_bricks._target = None
    else:
        new_bricks._target = _axes_dict[brick1._label]
    
    for label in labels_all:
        if "_case" in _axes_dict[label].__dict__:
            _axes_dict[label].case

    if target is not None: 
        new_bricks._case_labels = new_bricks._case_labels + parent._case_labels + brick2._case_labels
    else:
        new_bricks._case_labels = new_bricks._case_labels + brick1._case_labels + brick2._case_labels
    
    for label in labels_all:
        new_bricks._labels.add(label) 
    return new_bricks

def stack(bricks, margin=None, operator="|", equal_spacing=False): 
    """
    Stack multiple Brick(s) objects horizontally or vetically.

    Parameters
    ----------
    bricks : list of patchworklib.Brick(s) objects 
        List composed of Brick(s) objects. The list can include both Brick and Brick(s) 
        objects. 
    margin : float, default: None 
        Margin size of each Brick(s). If None, the `pw.param["margin"]` value would be 
        used as the margin size between the Brick(s) objects.
    operator : str ("|" or "/"), default: "|"
        Orientation of the arrangement for the given Brick(s) object.
        If this value is "|" or "/", the width/height of the object to be stacked will be 
        adjusted so that it will be aligned with previous one.
    equal_spacing : bool, defeault: False
        If True, each Brick/Bricks object is placed with equal spaces based on the inner corners. 
    
    Returns
    -------
    None

    """

    global param 
    global _axes_dict
    if margin is None:
        pass 
    else:
        original = param["margin"]
        param["margin"] = margin 
    
    base = bricks[0] 
    if operator == "|":
        for brick in bricks[1:]:
            base = base | brick
    
    elif operator == "/":
        for brick in bricks[1:]:
            base = vstack(base, brick, direction="b") 
            #base = base / brick
    
    elif operator == "+":
        for brick in bricks[1:]:
            base = base + brick
    
    elif operator == "-":
        for brick in bricks[1:]:
            base = vstack(base, brick, adjust_width=False, direction="b") 
    
    else:
        raise ValueError("Please specify a operator from ('+', '-', '/', '-').") 

    if equal_spacing == True:
        spaces    = [] 
        if operator == "|" or operator == "+":
            for i, brick in enumerate(bricks[:-1]):
                x00, x01, y00, y01 = brick.get_inner_corner() 
                x10, x11, y10, y11 = bricks[i+1].get_inner_corner() 
                spaces.append(abs(x10-x01)) 
        else:
            for i, brick in enumerate(bricks[:-1]):
                x00, x01, y00, y01 = brick.get_inner_corner() 
                x10, x11, y10, y11 = bricks[i+1].get_inner_corner() 
                spaces.append(abs(y00-y11)) 
        
        max_space = max(spaces) 
        if operator == "|" or operator == "+": 
            first = bricks[0] 
            for second in bricks[1:]:
                x00, x01, y00, y01 = first.get_inner_corner() 
                x10, x11, y10, y11 = second.get_inner_corner()
                for ax in second.bricks_dict.values():
                    x20, x21, y20, y21 = ax.get_inner_corner()
                    ax.set_position([x01 + max_space + x20 - x10, y20, abs(x20-x21), abs(y20-y21)])
                first = second
        else:
            first = bricks[0] 
            for second in bricks[1:]:
                x00, x01, y00, y01 = first.get_inner_corner() 
                x10, x11, y10, y11 = second.get_inner_corner()
                for ax in second.bricks_dict.values():
                    x20, x21, y20, y21 = ax.get_inner_corner()
                    ax.set_position([x20, y00 - (max_space + abs(y11-y21)) - abs(y20-y21), abs(x20-x21), abs(y20-y21)])
                first = second
        
        base.set_originalpositions()

    if margin is None:
        pass
    else:
        param["margin"] = original 
    
    return base

class Bricks():
    """
    
    A patchworklib.Bricks class object is a collection of patchworklib.Brick object.
    It can also be joined with other Brick and Bricks object by using '/' 
    and '|' operators.

    """
    
    _num = 0
    def __init__(self, bricks_dict=None, label=None): 
        """
       
        Parameters
        ----------
        bricks_dict : dict
            Dictionaly of patchworklib.Brick class objects. The label name of each Brick 
            object is served as the dictionaly keys. 
        label : str
            Unique identifier for the Bricks class object. The value can be used in layout 
            adjustment using label indexing. The value would be assigned to `self.label`.
        
        Attributes
        ----------
        case : matplotlib.Axes.axes
            Invisible axes object surrounding Bricks object excluding common label, legend.
        outline : patchworklib.Bricks
            New bricks object based on the invisible axes object surrounding all objects in 
            the original Bricks object including `case` axes.
        label : str
            Unique identifier of the Bricks class object. If the Bricks object is 
            incorporated in the other super Bricks objects, by specifying the label name for 
            the super object as `Bricks_object[{label}]`, the Bricks object can be accessed from 
            the super Bricks object.
        bricks_dict : dict
            Dictionary with labels of the Brick objects in the Bricks object as dictionary keys 
            and the corresponding Brick objects as dictionary values.
        """

        global _axes_dict
        global _basefigure
        if label is None:
            self._label = "Bricks-" + str(Bricks._num)
        else:
            self._label = label
        
        self._labels = set([]) 
        for key in bricks_dict.keys():
            if type(bricks_dict[key]) in (Bricks, Brick, cBrick):
                pass 
            else:
                bricks_dict[key] = Brick(bricks_dict[key]) 
            self._labels.add(bricks_dict[key]._label)
        self._labels.add(self._label)

        _axes_dict[self._label] = self
        _bricks_list.append(self) 
        self.bricks_dict = bricks_dict 
        self._type  = "Bricks"
        Bricks._num += 1
        self._case = _basefigure.add_axes([0,0,1,1], label="case:" + self._label)
        x0, x1, y0, y1 = self.get_middle_corner() 
        self._case.set_position([x0, y0, x1-x0, y1-y0])
        self._case.patch.set_facecolor("#FFFFFF") 
        self._case.patch.set_alpha(0.0) 
        self._case.spines["right"].set_visible(False)   
        self._case.spines["top"].set_visible(False) 
        self._case.spines["bottom"].set_visible(False) 
        self._case.spines["left"].set_visible(False) 
        self._case.set_xticks([]) 
        self._case.set_yticks([])
        _axes_dict[self._case.get_label()] = self._case 
        self._case_labels   = [self._case.get_label()]  
        self._parent = None
        x0, x1, y0, y1 = self.get_outer_corner() 
        self._originalsize = (abs(x1-x0), abs(y0-y1))
        self.set_originalpositions() 

    def __getitem__(self, item):
        global _axes_dict
        if type(item) == Bricks or type(item) == Brick:
            item = item.get_label()
        
        if type(item) == str:
            if item in self.bricks_dict:
                self.bricks_dict[item]._parent = self._label
                return self.bricks_dict[item]

            elif item in _axes_dict and item in self._labels:
                _axes_dict[item]._parent = self._label
                return _axes_dict[item]

        elif type(item) == tuple:
            new_bricks_dict = {} 
            for key in item:
                if type(key) == str:
                    new_bricks_dict[key] = self.bricks_dict[key] 
                elif type(key) in (Bricks, Brick, cBrick): 
                    new_bricks_dict[key] = self.bricks_dict[key.get_label()] 
            new_bricks = Bricks(bricks_dict=new_bricks_dict) 
            new_bricks._parent = self._label
            return new_bricks 

    def __getattribute__(self, name):
        if name == "case":
            x0, x1, y0, y1 = self.get_middle_corner() 
            pos = self._case.get_position() 
            px0, px1, py0, py1 = pos.x0, pos.x1, pos.y0, pos.y1
            if (_round_it(x0,4), _round_it(x1,4), _round_it(y0,4), _round_it(y1,4)) == (_round_it(px0,4), _round_it(px1,4), _round_it(py0,4), _round_it(py1,4)):
                pass 
            else:
                self._case.set_position([x0, y0, x1-x0, y1-y0])
                _reset_ggplot_legend(self)
            return self._case
        
        elif name == "label":
            return self._label
        
        elif name == "outline":
            labels_all = self._labels 
            for label in labels_all:
                if "_case" in _axes_dict[label].__dict__:
                    _axes_dict[label].case
            
            x0, x1, y0, y1 = self.get_outer_corner() 
            new_dict = {} 
            for key in self.bricks_dict:
                new_dict[key] = self.bricks_dict[key]
            outline_label = "outline:{}".format(self._label)
            if outline_label in pBrick._labelset:
                ax = _axes_dict[outline_label]
            else:
                ax = Brick(label=outline_label) 
            ax.set_position([x0, y0, x1-x0, y1-y0]) 
            ax.patch.set_facecolor("#FFFFFF") 
            ax.patch.set_alpha(0.0) 
            ax.spines["right"].set_visible(False)   
            ax.spines["top"].set_visible(False) 
            ax.spines["bottom"].set_visible(False) 
            ax.spines["left"].set_visible(False) 
            ax.set_xticks([]) 
            ax.set_yticks([])
            new_dict[outline_label] = ax
            bricks = Bricks(bricks_dict=new_dict, label="Bricks-"+outline_label)  
            bricks._case_labels = list(set(bricks._case_labels + self._case_labels))
            bricks._labels = (bricks._labels | self._labels) 
            return bricks
        
        else:
            return super().__getattribute__(name) 

    def _comeback(self): 
        global _removed_axes
        global _basefigure
        fig  = _basefigure
        for label, ax in _removed_axes.items():
            ax.figure = fig
            fig.add_axes(ax)
        _removed_axes = {}
        
    def set_originalpositions(self):
        self._originalpositions = {}
        labels_all = set(self._labels) | set(self._case_labels)
        for label in labels_all:
            if type(_axes_dict[label]) == Brick: #or type(_axes_dict[label]) == axes.Axes:
                self._originalpositions[label] = _axes_dict[label].get_inner_corner()
            else:
                pass 

    def reset_label(self, name):
        """

        Reset the label value of itself with `name`. 
        
        Parameters
        ----------
        name : str
            New name used as `self._label`.
        
        """

        global _axes_dict
        _axes_dict[name] = _axes_dict.pop(self._label) 
        self._labels = self._labels - set([self._label])  
        self._case_labels.remove("case:" + self._label)  
        self._case_labels.append("case:" + name) 
        self._case._label = "case:" + name
        self._label  = name 
        self._labels.add(name) 
    
    def get_label(self):
        """
        
        Get and return the value of `self._label`. 
        
        """

        return self._label
    
    def change_plotsize(self, new_size):
        """

        Change the plot sizes in the Bricks object.
     
        Parameters
        ----------
        new_size : tuple (float, float) 

        """
        self._comeback() 
        outers = self.get_outer_corner()  
        expand(self.outline, new_size[0]/abs(outers[0]-outers[1]), new_size[1]/abs(outers[3]-outers[2]))  
        x0, x1, y0, y1     = self.get_outer_corner() 
        self._originalsize = (abs(x1-x0), abs(y0-y1))
        self.set_originalpositions() 
        self.case

    def align_xlabels(self, keys=None):
        global _basefigure
        global _axes_dict
        renderer = _basefigure.canvas.get_renderer()
        
        if keys is None:
            keys = [key for key in self._labels if key[0:5] != "case:" and key[0:8] != "outline:"]
        else: 
            pass 
        targets = [key if type(key) in (Bricks, Brick, cBrick) else _axes_dict[key] for key in keys] 

        miny = None
        for target in targets:
            if type(target) == Bricks:
                x, y = target._case.xaxis.get_label().get_position()  
                text = target._case.xaxis.get_label().get_text()  
            else:
                x, y = target.xaxis.get_label().get_position()  
                text = target.xaxis.get_label().get_text()  
            if miny is None or y < miny:
                miny = y 

        for target in targets:
            if type(target) != Bricks:
                pad  = target.xaxis.labelpad
                x, y = target.xaxis.get_label().get_position()  
                text = target.xaxis.get_label().get_text()  
                target.set_xlabel(text, x=x, y=y, labelpad=pad + (y-miny)*(72 / _basefigure.dpi))   
            else:
                pad  = target._case.xaxis.labelpad
                x, y = target._case.xaxis.get_label().get_position()  
                text = target._case.xaxis.get_label().get_text()  
                target._case.set_xlabel(text, x=x, y=y, labelpad=pad + (y-miny)*(72 / _basefigure.dpi))   

    def align_ylabels(self, keys=None, ha="left"): 
        global _basefigure
        global _axes_dict
        renderer = _basefigure.canvas.get_renderer()
        
        if keys is None:
            keys = [key for key in self._labels if key[0:5] != "case:" and key[0:8] != "outline:"]
        else: 
            pass 
        targets = [key if type(key) in (Bricks, Brick, cBrick) else _axes_dict[key] for key in keys] 

        minx = None
        for target in targets:
            if type(target) == Bricks:
                x, y = target._case.yaxis.get_label().get_position()  
                text = target._case.yaxis.get_label().get_text()  
            else:
                x, y = target.yaxis.get_label().get_position()  
                text = target.yaxis.get_label().get_text()  
            if minx is None or x < minx:
                minx = x 

        for target in targets:
            if type(target) != Bricks:
                pad  = target.yaxis.labelpad
                x, y = target.yaxis.get_label().get_position()  
                text = target.yaxis.get_label().get_text()  
                target.set_ylabel(text, x=x, y=y, labelpad=pad + (x-minx)*(72 / _basefigure.dpi))   
            else:
                pad  = target._case.yaxis.labelpad
                x, y = target._case.yaxis.get_label().get_position()  
                text = target._case.yaxis.get_label().get_text()  
                target._case.set_ylabel(text, x=x, y=y, labelpad=pad + (x-minx)*(72 / _basefigure.dpi))   

    def set_supxlabel(self, xlabel, labelpad=None, *, loc=None, **args):
        """

        Set a common xlabel for the Brick(s) objects in the Bricks object.
        A Bricks class object is composed of multiple Brick class objects and they 
        sometimes share common xaxis and yaxis labels.For such a case, you can remove 
        redundant labels to be drawn on the figure and can add common x axis label for 
        all Brick(s) objects in the Bricks object. Actually, this function is the 
        wrapper function of `self.case.set_xlabel`. 

        Parameters
        ----------
        xlabel : str
            xlabel value 
        labelpad : int, default: 8
            Spacing in points from the virtual axes bounding box of the Bricks object.
        args : dict
            Text properties control the appearance of the label.

        Returns
        -------
        matplotlib.text.Text
        
        """

        if labelpad is None:
            labelpad = matplotlib.rcParams["axes.labelpad"] + 4
        mx0, mx1, my0, my1 = self.get_middle_corner() 
        ix0, ix1, iy0, iy1 = self.get_inner_corner() 
        args.setdefault("x", ((abs(ix0-mx0) + abs(ix1-mx0))/2)/abs(mx1-mx0))
        return self._case.set_xlabel(xlabel, labelpad=labelpad, loc=loc, **args) 

    def set_supxlabel_positon(self, position):
        self._case.xaxis.set_label_position(position) 
    
    def set_supylabel(self, ylabel, labelpad=None, *, loc=None, **args):
        """

        Set a common ylabel for the Brick(s) objects in the Bricks object.
        A Bricks class object is composed of multiple Brick class objects and they 
        sometimes share common xaxis and yaxis labels. For such a case, you can remove 
        redundant labels to be drawn on the figure and can add common y axis label for 
        all Brick(s) objects in the Bricks object. Actually, this function is the 
        wrapper function of `self.case.set_ylabel`.
        
        Parameters
        ----------
        ylabel : str
            ylabel value 
        
        labelpad : int, default: 8
            Spacing in points from the virtual axes bounding box of the Bricks object.
        
        args : dict
            Text properties control the appearance of the label.

        Returns
        -------
        matplotlib.text.Text
        
        """

        if labelpad is None:
            labelpad = matplotlib.rcParams["axes.labelpad"] + 4
        mx0, mx1, my0, my1 = self.get_middle_corner() 
        ix0, ix1, iy0, iy1 = self.get_inner_corner() 
        args.setdefault("y", ((abs(iy0-my0) + abs(iy1-my0))/2)/abs(my1-my0))
        return self._case.set_ylabel(ylabel, labelpad=labelpad, loc=loc, **args)

    def set_supylabel_positon(self, position):
        self._case.yaxis.set_label_position(position) 

    def set_suptitle(self, title, loc=None, pad=None, **args):
        """

        Set a common title for the Brick(s) objects in the Bricks object.
        A Bricks class object is composed of multiple Brick class objects and they 
        sometimes share common xaxis and yaxis labels. For such a case, you can set 
        common title for all Brick(s) objects in the Bricks object. Actually, this 
        function is the wrapper function of `self.case.set_title`.
        
        Parameters
        ----------
        title : str
            title value 
        loc : str ("center", "left", "right), default "center"
            Which title to set.
        pad : int, default: 12
            Spacing in points from the virtual axes bounding box of the Bricks object.
        args : dict
            Text properties control the appearance of the label.
        
        Returns
        -------
        matplotlib.text.Text
        
        """

        if pad is None:
            pad = matplotlib.rcParams["axes.labelpad"] + 8
        mx0, mx1, my0, my1 = self.get_middle_corner() 
        ix0, ix1, iy0, iy1 = self.get_inner_corner() 
        args.setdefault("x", ((abs(ix0-mx0) + abs(ix1-mx0))/2)/abs(mx1-mx0))
        args.setdefault("y", None)
        return self._case.set_title(title, pad=pad, loc=loc, **args)  
    
    def set_index(self, index, x=None, y=None, **args):
        """

        Set a index label on 'upper left' of the Bricks object.
        An index labels can be added, such as those on sub-figures published in 
        scientific journals. Actually, this function is the wrapper function of 
        `self.case.text`.

        Parameters
        ----------
        index : str
            index value 
        x : float
            By default, the value will be adjusted as index label is placed on 'upper left' 
            of the Bricks object. 
        y : flaot, 
            By default, the value will be adjusted as index label is placed on 'upper left' 
            of the Bricks object.
        args : dict
            Text properties control the appearance of the label.
        
        Returns
        -------
        matplotlib.text.Text
        
        """

        ox0, ox1, oy0, oy1 = self.get_outer_corner() 
        mx0, mx1, my0, my1 = self.get_middle_corner() 
        ix0, ix1, iy0, iy1 = self.get_inner_corner() 
        if x is None:
            x =  -1.01 * (abs(ox0-mx0))/abs(mx1-mx0)
        if y is None:
            y = 1.01 * abs(oy1-my0)/abs(my1-my0)
        return self._case.text(x, y, index, **args)  
    
    def set_text(self, x, y, text, **args):  
        return self._case.text(x, y, text, **args)

    def set_supspine(self, which="left", visible=True, position=None, bounds=None):
        """ 

        Set a common spine for the Bric(s) objects in the Bricks object.   
        The spines of `self.case` surrounding the Bricks object are invisible by default. 
        However, by applying this methods, a specified spine will be visible.

        Parameters
        ----------
        which : str ('left', 'right', 'top', 'bottom'), default: 'left'
            Kind of the spine 
        visible : bool, default: True
            Setting of Show/hide the spine
        position : tuple (position type ('outward', 'axes', 'data'), amount (float)), 
            Position of the spine. 
            For details, please see 'https://matplotlib.org/3.5.1/api/spines_api.html'.
        bounds : tuple (float, float), 
            Bounds of the spine. 
            For details, please see 'https://matplotlib.org/3.5.1/api/spines_api.html'.

        Returns
        -------
        matplotlib.spines.Spine

        """
        
        ix0, ix1, iy0, iy1 = self.get_inner_corner() 
        mx0, mx1, my0, my1 = self.get_middle_corner()
        
        if position is None:
            if which == "bottom":
                position = ("data", -0.1 / abs(my0-my1))
            
            elif which == "top":
                position = ("data", 1 + 0.1 / abs(my0-my1))
            
            elif which == "left":
                position = ("data", -0.1 / abs(mx0-mx1))
            
            elif which == "right":
                position = ("data", 1 + 0.1 / abs(mx0-mx1))
        self._case.spines[which].set_position(position)
        
        if bounds is None:
            if which == "bottom" or which == "top":
                low  = abs(ix0-mx0)/abs(mx1-mx0) 
                high = abs(ix1-mx0)/abs(mx1-mx0)
            
            elif which == "left" or which == "right":
                low  = abs(iy0-my0)/abs(my1-my0) 
                high = abs(iy1-my0)/abs(my1-my0)
            
            bounds = (low, high) 
        self._case.spines[which].set_bounds(*bounds)
        
        self._case.spines[which].set_visible(visible) 
        return self._case.spines[which]
    
    
    def add_colorbar(self, cmap=None, x=None, y=None, vmin=0, vmax=1, hratio=None, wratio=None, coordinate="relative", label=None, **args):
        """

        Set a common colorbar for Brick(s) objects in the Bricks object and return 
        a new Bricks object including the colorbar.
        
        Parameters
        ----------
        cmap : Colormap, default: 'viridis'
            The colormap to use.
        x : float, default: None
            if args['orientation'] is 'vertical', the value will be adjusted as the colorbar 
            is placed on 'lower right' of the Bricks object. if args['orientation'] is 
            'horizontal', the value will be adjusted as the colorbar is placed on 'lower center' 
            of the Bricks object. The zero position for `x` is the most left axes of the Brick 
            objects in the Bricks object.
        y : float, default: None
            if args['orientation'] is 'vertical', the value will be adjusted as the colorbar 
            is placed on 'lower right' of the Bricks object. if args['orientation'] is 
            'horizontal', the value will be adjusted as the colorbar is placed on 'lower center' 
            of the Bricks object. The zero position for `y` is the most bottom axes of the 
            Brick objects in the Bricks object.
        vmin : float, default: 0
            Minimum value to anchor the colormap.
        vmax : float, default: 1
            Maximum value to anchor the colormap.
        hratio : float 
            Height ratio of colorbar to height of self.case
        wratio : float 
            Width ratio of colorbar to width of self.case
        coordinate : str ("relative", "absolute"), default "relative"
            if "absolute", the values of x and y will mean the inches of the distances from the 
            base zero positions. if "relative", the values of x and y will mean the relative 
            distances based on width and height of Bricks object from the base zero positions.
          
          
        Returns
        -------
        patchworklib.Bricks object

        """

        ox0, ox1, oy0, oy1 = self.get_outer_corner() 
        mx0, mx1, my0, my1 = self.get_middle_corner() 
        ix0, ix1, iy0, iy1 = self.get_inner_corner() 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        
        args.setdefault("orientation", "vertical") 
        if args["orientation"] == "vertical":
            if hratio is None:
                hratio = 0.4
            height = abs(iy1-iy0)*hratio
            
            if wratio is None:
                wratio = 0.05 
                width  = abs(ix1-ix0)*wratio
                if width > 0.2:
                    width = 0.2
            else:
                width  = abs(ix1-ix0)*wratio

            ax = Brick(figsize=(width, height), label=label)  
            cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, **args)
            
            if x is None:
                xorigin = None
                if coordinate == "relative":
                    x = (1.0 + (abs(mx1-ix1) / abs(ix0-ix1)) + 0.05) * abs(ix0-ix1) 

                if coordinate == "absolute":
                    x = 0.3 + mx1 - ix0
            else:
                xorigin = x 
                if coordinate == "relative":
                    x = x * abs(ix0-ix1) 
                else: 
                    pass 

            if y is None:
                yorigin = None
                y = 0
            else:
                yorigin = y 
                if coordinate == "relative":
                    y =  y * abs(iy0-iy1)
                else:
                    pass 

        else:
            if wratio is None:
                wratio = 0.4
            width  = abs(ix1-ix0)*wratio
            
            if hratio is None:
                hratio = 0.05 
                height = abs(iy1-iy0)*hratio
                if height > 0.2:
                    height = 0.2
            else:
                height = abs(iy1-iy0)*hratio

            ax = Brick(figsize=(width, height), label=label)  
            cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, **args)
            axx0, axx1, axy0, axy1 = ax.get_outer_corner() 
            if x is None:
                xorigin = None
                if coordinate == "relative": 
                    x = (0.5 - (abs(axx0-axx1)/abs(ix0-ix1)) * 0.5) * abs(ix0-ix1)
                
                if coordinate == "absolute":
                    x = (ix1+ix0)/2  - abs(axx0-axx1)*0.5 - ix0
            else:
                xorigin = x
                if coordinate == "relative":
                    x = x * abs(ix0-ix1) 
                else: 
                    pass 

            if y is None:
                yorigin = None
                if coordinate == "relative":
                    y = (-1.0 * (abs(my0-iy0) / abs(iy0-iy1)) - 0.05) * abs(iy0-iy1)
                
                if coordinate == "absolute": 
                    y = iy0 - abs(my0-iy0) - iy0 - 0.4
            else:
                yorigin = y
                if coordinate == "relative":
                    y =  y * abs(iy0-iy1)
                else:
                    pass 

        if args["orientation"] == "vertical":
            ax.set_position([ix0 + x, iy0 + y, width, height])  
        else:   
            ax.set_position([ix0 + x, iy0 + y - height, width, height])  

        new_bricks_dict = {} 
        for key in self.bricks_dict:
            new_bricks_dict[key]   = self.bricks_dict[key]
        new_bricks_dict[ax._label] = ax 
        new_bricks = Bricks(bricks_dict = new_bricks_dict) 
        new_bricks._case_labels = list(set(new_bricks._case_labels + self._case_labels + ax._case_labels))
        new_bricks._labels = (new_bricks._labels | self._labels | ax._labels) 
        return new_bricks

    def move_legend(self, new_loc, **kws):
        """

        Move legend
        
        Parameters
        ----------
        new_lock : str or int
            Location argument, as in matplotlib.axes.Axes.legend().
        kws : dict
            Other keyword arguments can be used in matplotlib.axes.Axes.legend().

        """
        old_legend = self._case.legend_
        
        if (matplotlib.__version__) >= StrictVersion("3.8"):
            handles = old_legend.legend_handles
        else:
            handles = old_legend.legendHandles

        labels     = [t.get_text() for t in old_legend.get_texts()]
        title      = old_legend.get_title().get_text()
        self._seaborn_legend[0]["loc"] = new_loc
        for key in kws:
            self._seaborn_legend[0][key] = kws[key]  
        if "bbox_to_anchor" in kws:
            self._seaborn_legend = (self._seaborn_legend[0], kws["bbox_to_anchor"])
        else:
            self._seaborn_legend = (self._seaborn_legend[0], None) 
        _reset_ggplot_legend(self)
        #self.case

    def get_inner_corner(self, labels=None):
        """

        Return the most left, right, bottom, and top positions of the Brick objects in the Bricks object.
        patchworklib.Bricks object is a collection of patchworklib.Brick object. 
        Here, the inner corners of the Bricks object means the most top left, top right, 
        bottom left, and bottom right corners of the spines of Brick objects in the Bricks object.
        
        Returns
        -------
        tuple (left, right, bottom, top) 

        """
        x0_list = [] 
        x1_list = [] 
        y0_list = [] 
        y1_list = [] 
        if labels is None:
            labels = set(self.bricks_dict.keys()) 
        for key in self.bricks_dict:
            if key in labels:
                ax  = self.bricks_dict[key]  
                pos = ax.get_position()  
                x0_list.append(pos.x0) 
                x1_list.append(pos.x1)
                y0_list.append(pos.y0) 
                y1_list.append(pos.y1) 
        return min(x0_list), max(x1_list), min(y0_list), max(y1_list) 

    def get_middle_corner(self, labels=None): 
        """

        Return the left, right, bottom, and top positions of `self.case`. 
        patchworklib.Bricks object provides `case` attribute. `case` is invisible matplotlib.axes.Axes object 
        surrounding Brick objects in the Bricks object and their artist, text objects. 
        `get_middle_corner` method returns the corner position of `case`.
        
        Returns
        -------
        tuple (left, right, bottom, top) 

        """
        global _basefigure
        global _render
        global _scale
        x0_list = [] 
        x1_list = [] 
        y0_list = [] 
        y1_list = [] 
        if labels is None:
            labels = set(self.bricks_dict.keys()) 
        
        for key in self.bricks_dict:
            if key in labels:
                ax   = self.bricks_dict[key]  
                px0, px1, py0, py1 = ax.get_middle_corner()
                x0_list.append(px0) 
                x1_list.append(px1)
                y0_list.append(py0) 
                y1_list.append(py1) 
        return min(x0_list), max(x1_list), min(y0_list), max(y1_list) 
    
    def get_outer_corner(self):
        """

        Return the left, right, bottom, and top positions of `self.outline`. 
        patchworklib.Bricks object provides `outline` attribute. `outline` is Bricks object based on 
        invisible matplotlib.axes.Axes object surrounding `case` axes, and its artist, text objects. 
        `get_outer_corner` method returns the corner position of `outline`.
        
        Returns
        -------
        tuple (left, right, bottom, top) 

        """
        global _basefigure 
        global _axes_dict
        global _render
        global _scale
        x0_list = [] 
        x1_list = [] 
        y0_list = [] 
        y1_list = []
        
        used_labels = set([])
        for key in self.bricks_dict:
            ax   = self.bricks_dict[key]  
            px0, px1, py0, py1 = ax.get_outer_corner()  
            x0_list.append(px0) 
            x1_list.append(px1)
            y0_list.append(py0) 
            y1_list.append(py1)
            for caselabel in ax._case_labels:
                used_labels.add(caselabel)

        for caselabel in self._case_labels:
            if caselabel in used_labels:
                pass 
            else:
                h, v = _basefigure.get_size_inches()
                pos  = _axes_dict[caselabel[5:]].case.get_tightbbox(_render)
                pos  = TransformedBbox(pos, _scale)
                x0_list.append(pos.x0/h) 
                x1_list.append(pos.x1/h)
                y0_list.append(pos.y0/v) 
                y1_list.append(pos.y1/v)  
        
        return min(x0_list), max(x1_list), min(y0_list), max(y1_list)
    
    def savefig(self, fname=None, transparent=None, quick=True, _ggplot=False, **kwargs):
        """
        
        Save figure. 
        The method is implemented based on the function of `matplotlib.pyplot.savefig`. 
        Therefore,same paraemters can be used.
        For detail, please see https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.savefig.html 

        Note
        ----
        1. If you face some problemes when using the method. please set `quick` parameter as `False`. 
        2. If the Bricks object is composed of more than a few dozens of Brick objects, 
        the methods consume a lot of times for saving the figure..., please take a coffe break.

        """
        for label in self._originalpositions:
            x0, x1, y0, y1 = self._originalpositions[label]
            _axes_dict[label].set_position([x0, y0, x1-x0, y1-y0])
        for case_label in self._case_labels:
            _axes_dict[":".join(case_label.split(":")[1:])].case 

        global param
        global _basefigure
        global _removed_axes
        if quick == False:
            self.case
            bytefig = io.BytesIO()  
            key0 = list(self.bricks_dict.keys())[0] 
            dill.dump(self.bricks_dict[key0].__class__._figure, bytefig)
            bytefig.seek(0) 
            tmpfig = dill.load(bytefig) 
            
            for ax in tmpfig.axes:
                if ax.get_label() in self.bricks_dict or ax.get_label() in self._case_labels:
                    pass 
                else:
                    ax.remove()

            if fname is not None: 
                kwargs.setdefault('bbox_inches', 'tight')
                kwargs.setdefault('dpi', param['dpi'])
                tmpfig.savefig(fname, transparent=transparent, **kwargs)
            return tmpfig 

        else:
            key0 = list(self.bricks_dict.keys())[0]  
            fig  = self.bricks_dict[key0].__class__._figure
            for label, ax in _removed_axes.items():
                ax.figure = fig
                fig.add_axes(ax)
            
            _removed_axes = {}    
            for ax in fig.axes:
                if ax.get_label() in self.bricks_dict or ax.get_label() in self._case_labels:
                    pass 
                else:
                    ax.remove()
                    _removed_axes[ax.get_label()] = ax

            if fname is not None: 
                kwargs.setdefault('bbox_inches', 'tight')
                kwargs.setdefault('dpi', param['dpi'])
                fig.savefig(fname, transparent=transparent, **kwargs) 
            else:
                if _ggplot == True:
                    bytefig = io.BytesIO()  
                    _basefigure.savefig(bytefig, format="pdf")
                 
            return fig 
    
    def __or__(self, other):
        self._comeback()
        other._comeback()
        if other._type == "spacer":
            return other.__ror__(self) 

        elif self._parent is not None:
            if other._parent is not None: #other._type == "Brick" and other._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            return hstack(_axes_dict[self._parent], other, target=self)
        else:
            if other._parent is not None: #other._type == "Brick" and other._parent is not None: #ax1 | ax23[3]
                return hstack(_axes_dict[other._parent], self, target=other, direction="l")
            else:
                return hstack(self, other) 
    
    def __add__(self, other):
        self._comeback()
        other._comeback()
        if other._type == "spacer":
            return other.__ror__(self) 
        elif self._parent is not None:
            if other._parent is not None: #other._type == "Brick" and other._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            return hstack(_axes_dict[self._parent], other, target=self)
        else:
            if other._parent is not None: #other._type == "Brick" and other._parent is not None: #ax1 | ax23[3]
                return hstack(_axes_dict[other._parent], self, target=other, direction="l")
            else:
                return hstack(self, other, adjust_height=False) 

    def __truediv__(self, other):
        self._comeback() 
        other._comeback()
        if other._type == "spacer":
            return other.__rtruediv__(self) 

        elif other._parent is not None: #other._type == "Brick" and other._parent is not None:
            if self._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            else:
                return vstack(_axes_dict[other._parent], self, target=other)
        else:
            if self._parent is not None:
                return vstack(_axes_dict[self._parent], other, target=self, direction="b")
            else:
                return vstack(other, self)
    
    def __sub__(self, other):
        self._comeback() 
        other._comeback()
        if other._type == "spacer":
            return other.__rtruediv__(self) 

        elif other._parent is not None: #other._type == "Brick" and other._parent is not None:
            if self._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            else:
                return vstack(_axes_dict[other._parent], self, target=other)
        else:
            if self._parent is not None:
                return vstack(_axes_dict[self._parent], other, target=self, direction="b")
            else:
                return vstack(other, self, adjust_width=False)

    def _repr_png_(self):
        buf = io.BytesIO()
        self.savefig(buf, "png")
        return buf.getvalue()

    def _repr_pdf_(self):
        buf = io.BytesIO()
        self.savefig(buf, "pdf")
        return buf.getvalue()

#class Brick(axes.Axes): 
class pBrick: 
    """

    Subclass of matplotlib.axes.Axes object.
    A Brick class object can be joined with another Brick and Bricks object 
    by using '/' and '|' operators.
    
    """
    
    axnum     = 0    
    _figure   = _basefigure
    _labelset = set([]) 
    
    def set_position(self, position):
        self._middle_flag = False
        self._outer_flag  = False
        super().set_position(position)

    def reset_label(self, name):
        """

        Reset the label value of itself with `name`. 
        
        Parameters
        ----------
        name : str
            New name used as `self._label`.

        """

        global _axes_dict
        global _bricks_list
        _axes_dict[name] = _axes_dict.pop(self._label) 
        for bricks  in bricks_list:
            if self._label in bricks.bricks_dict: 
                bricks.bricks_dict[name] =  bricks.bricks_dict.pop(self._label) 
        self._case_labels.remove("case:" + self._label)  
        self._case_labels.append("case:" + name) 
        self._case._label = "case:" + name
        self._labels    = self._labels - set([self._label])  
        pBrick._labelset = pBrick._labelset - set([self._label])  
        self._label = name
    
    def set_index(self, index, x=None, y=None, **args):
        """

        Set a index label on 'upper left' of the Bricks object.
        An index labels can be added, such as those on sub-figures published in scientific journals.
        Actually, this function is the wrapper function of `self.case.text`.

        Parameters
        ----------
        index : str
            index value 
        x : float,  
            By default, the value will be adjusted as index label is placed on 'upper left' 
            of the Bricks object. 
        y : flaot, 
            By default, the value will be adjusted as index label is placed on 'upper left' 
            of the Bricks object.
        args : dict
            Text properties control the appearance of the label.
   
        Returns
        -------
        matplotlib.text.Text
        
        """
        ox0, ox1, oy0, oy1 = self.get_outer_corner() 
        mx0, mx1, my0, my1 = self.get_middle_corner() 
        ix0, ix1, iy0, iy1 = self.get_inner_corner() 
        if x is None:
            x =  -1 * (abs(ox0-mx0))/abs(mx1-mx0)
        if y is None:
            y = abs(oy1-my0)/abs(my1-my0)
        return self._case.text(x, y, index, **args)  
    
    def add_text(self, x, y, text, **args):  
        self._middle_flag = False
        self._outer_flag  = False
        return self._case.text(x, y, text, **args)
    
    def add_colorbar(self, cmap=None, x=None, y=None, vmin=0, vmax=1, hratio=None, wratio=None, coordinate="relative", label=None, **args):
        """
        
        Set a common colorbar for Brick(s) objects in the Bricks object and return 
        a new Bricks object including the colorbar.

        Parameters
        ----------
        cmap : Colormap, default: 'viridis'
            The colormap to use.
        x : float, default: None
            if args['orientation'] is 'vertical', the value will be adjusted as the colorbar 
            is placed on 'lower right' of the Bricks object. if args['orientation'] is 
            'horizontal', the value will be adjusted as the colorbar is placed on 'lower center' 
            of the Bricks object. The zero position for `x` is the most left axes of the Brick 
            objects in the Bricks object.
        y : float, default: None
            if args['orientation'] is 'vertical', the value will be adjusted as the colorbar 
            is placed on 'lower right' of the Bricks object. if args['orientation'] is 
            'horizontal', the value will be adjusted as the colorbar is placed on 'lower center' 
            of the Bricks object. The zero position for `y` is the most bottom axes of the 
            Brick objects in the Bricks object.
        vmin : float, default: 0
            Minimum value to anchor the colormap.
        vmax : float, default: 1
            Maximum value to anchor the colormap.
        hratio : float 
            Height ratio of colorbar to height of self.case
        wratio : float 
            Width ratio of colorbar to width of self.case
        coordinate : str ("relative", "absolute"), default "relative"
            if "absolute", the values of x and y will mean the inches of the distances from the 
            base zero positions. if "relative", the values of x and y will mean the relative 
            distances based on width and height of Bricks object from the base zero positions.

        Returns
        -------
        patchworklib.Bricks object

        """

        ox0, ox1, oy0, oy1 = self.get_outer_corner() 
        mx0, mx1, my0, my1 = self.get_middle_corner() 
        ix0, ix1, iy0, iy1 = self.get_inner_corner() 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        
        args.setdefault("orientation", "vertical") 
        if args["orientation"] == "vertical":
            if hratio is None:
                hratio = 0.4
            height = abs(iy1-iy0)*hratio
            
            if wratio is None:
                wratio = 0.05 
                width  = abs(ix1-ix0)*wratio
                if width > 0.2:
                    width = 0.2
            else:
                width  = abs(ix1-ix0)*wratio

            ax = Brick(figsize=(width, height), label=label)  
            cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, **args)
            
            if x is None:
                if coordinate == "relative":
                    x = (1.0 + (abs(mx1-ix1) / abs(ix0-ix1)) + 0.05) * abs(ix0-ix1) 

                if coordinate == "absolute":
                    x = 0.3 + mx1 - ix0
            else:
                if coordinate == "relative":
                    x = x * abs(ix0-ix1) 
                else: 
                    pass 

            if y is None:
                y = 0
            
            else:
                if coordinate == "relative":
                    y =  y * abs(iy0-iy1)
                else:
                    pass 

        else:
            if wratio is None:
                wratio = 0.4
            width  = abs(ix1-ix0)*wratio
            
            if hratio is None:
                hratio = 0.05 
                height = abs(iy1-iy0)*hratio
                if height > 0.2:
                    height = 0.2
            else:
                height = abs(iy1-iy0)*hratio

            ax = Brick(figsize=(width, height), label=label)  
            cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, **args)
            axx0, axx1, axy0, axy1 = ax.get_outer_corner() 
            if x is None:
                if coordinate == "relative": 
                    x = (0.5 - (abs(axx0-axx1)/abs(ix0-ix1)) * 0.5) * abs(ix0-ix1)
                
                if coordinate == "absolute":
                    x = (ix1+ix0)/2  - abs(axx0-axx1)*0.5 - ix0
            else:
                if coordinate == "relative":
                    x = x * abs(ix0-ix1) 
                else: 
                    pass 
            

            if y is None:
                if coordinate == "relative":
                    y = (-1.0 * (abs(my0-iy0) / abs(iy0-iy1)) - 0.05) * abs(iy0-iy1)
                
                if coordinate == "absolute": 
                    y = iy0 - abs(my0-iy0) - iy0 - 0.4
            else:
                if coordinate == "relative":
                    y =  y * abs(iy0-iy1)
                else:
                    pass 
        
        if args["orientation"] == "vertical":
            ax.set_position([ix0 + x, iy0 + y, width, height])  
        else:    
            ax.set_position([ix0 + x, iy0 + y - height, width, height])  

        new_bricks_dict = {} 
        for key in self.bricks_dict:
            new_bricks_dict[key]   = self.bricks_dict[key]
        new_bricks_dict[ax._label] = ax 
        new_bricks = Bricks(bricks_dict = new_bricks_dict) 
        new_bricks._case_labels = list(set(new_bricks._case_labels + self._case_labels + ax._case_labels))
        new_bricks._labels = (new_bricks._labels | self._labels | ax._labels) 
        self._middle_flag = False
        self._outer_flag  = False
        return new_bricks

    def _comeback(self): 
        global _removed_axes
        global _basefigure
        fig  = _basefigure
        for label, ax in _removed_axes.items():
            ax.figure = fig
            fig.add_axes(ax)
        _removed_axes = {}

    def get_inner_corner(self, labels=None):
        """

        Return the left, right, bottom, and top positions of the Brick object.
        
        """

        self._comeback() 
        pos = self.get_position()  
        return pos.x0, pos.x1, pos.y0, pos.y1

    def get_middle_corner(self, labels=None):
        """

        Return the left, right, bottom, and top positions of `self.case`. 
        
        """
        global _render 
        global _scale 
        global _basefigure
        self._comeback()
        if self._middle_flag == True:
            #print("pass") 
            return self._middle_corner 
        else:
            h, v = _basefigure.get_size_inches()
            pos  = self.get_tightbbox(_render)
            pos  = TransformedBbox(pos, _scale)
            self._middle_corner = pos.x0/h, pos.x1/h, pos.y0/v, pos.y1/v 
            self._middle_flag = True
            return self._middle_corner
    
    def get_outer_corner(self, labes=None): 
        """

        Return the left, right, bottom, and top positions of `self.outline`. 
        patchworklib.Bricks object provides `outline` attribute. `outline` is Bricks object based on 
        invisible matplotlib.axes.Axes object surrounding `case` axes, and its artist, text objects. 
        `get_outer_corner` method returns the corner position of `outline`.
        In generally, it will return same result with `get_middle_corner`.
        
        Returns
        -------
        tuple (left, right, bottom, top) 

        """
        global _render
        global _scale
        global _basefigure
        self._comeback()
        
        if self._outer_flag == True:
            #print("pass")
            return self._outer_corner
        else:
            h, v = _basefigure.get_size_inches()
            pos1 = self.get_tightbbox(_render)
            pos1 = TransformedBbox(pos1, _scale)

            pos2 = self.case.get_tightbbox(_render)
            pos2 = TransformedBbox(pos2, _scale)
            self._outer_corner = min([pos1.x0/h, pos2.x0/h]), max([pos1.x1/h, pos2.x1/h]), min([pos1.y0/v, pos2.y0/v]), max([pos1.y1/v, pos2.y1/v])
            self._outer_flag = True
            return self._outer_corner

    def savefig(self, fname=None, transparent=None, quick=True, _ggplot=False, **kwargs):
        """

        Save figure.
        The method is implemented based on the function of `matplotlib.pyplot.savefig`. 
        Therefore,same paraemters can be used.
        For detail, please see https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.savefig.html 

        """
        
        global param
        global _removed_axes
        self.set_position([0, 0, self._originalsize[0], self._originalsize[1]]) 

        if quick == False:
            self.case
            bytefig = io.BytesIO()  
            key0 = list(self.bricks_dict.keys())[0] 
            dill.dump(self.bricks_dict[key0].__class__._figure, bytefig)
            bytefig.seek(0) 
            tmpfig = dill.load(bytefig) 
            
            for ax in tmpfig.axes:
                if ax.get_label() in self.bricks_dict or ax.get_label() in self._case_labels:
                    pass 
                else:
                    ax.remove()

            if fname is not None: 
                kwargs.setdefault('bbox_inches', 'tight')
                kwargs.setdefault('dpi', param['dpi'])
                tmpfig.savefig(fname, transparent=transparent, **kwargs)
            return tmpfig

        else:
            key0 = list(self.bricks_dict.keys())[0]  
            fig  = self.bricks_dict[key0].__class__._figure
            for label, ax in _removed_axes.items():
                ax.figure = fig
                fig.add_axes(ax)
            _removed_axes = {}
            for ax in fig.axes:
                if ax.get_label() in self.bricks_dict or ax.get_label() in self._case_labels:
                    pass 
                else:
                    ax.remove()
                    _removed_axes[ax.get_label()] = ax
            if fname is not None: 
                kwargs.setdefault('bbox_inches', 'tight')
                kwargs.setdefault('dpi', param['dpi'])
                fig.savefig(fname, transparent=transparent, **kwargs) 
            else:
                if _ggplot == True:
                    bytefig = io.BytesIO()  
                    _basefigure.savefig(bytefig, format="pdf")
            
            return fig 
    
    def change_plotsize(self, new_size): 
        """

        Change the plot size of the Brick object.
        
        Parameters
        ----------
        new_size : tuple (float, float) 

        """
        
        self._comeback()
        if type(new_size) ==  tuple or type(new_size) == list:
            self.set_position([0, 0, new_size[0], new_size[1]])
            self._originalsize = new_size 
        else:
            self.set_position([0, 0, 1, new_size])
            self._originalsize = (1, new_size) 
        _reset_ggplot_legend(self)
        self.case 
    
    def change_aspectratio(self, new_size):  
        """
        
        Change figsize
        This function is deprecated.
        
        """
        self.change_plotsize(new_size) 

    def move_legend(self, new_loc, **kws):
        """

        Move legend

        Parameters
        ----------
        new_loc : str or int
            Location argument, as in matplotlib.axes.Axes.legend().
        
        kws : dict
            Other keyword arguments can be used in matplotlib.axes.Axes.legend().

        """
        
        self._comeback()
        old_legend = self.legend_
        
        if (matplotlib.__version__) >= StrictVersion("3.8"):
            handles = old_legend.legend_handles
        else:
            handles = old_legend.legendHandles

        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        self._middle_flag = False
        self._outer_flag  = False
        self.legend(handles, labels, loc=new_loc, title=title, **kws)
    
    def __or__(self, other):
        self._comeback()
        other._comeback()
        if other._type == "spacer":
            return other.__ror__(self) 

        elif self._parent is not None:
            if other._parent is not None: #other._type == "Brick" and other._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            return hstack(_axes_dict[self._parent], other, target=self)
        else:
            if other._parent is not None: #other._type == "Brick" and other._parent is not None: #ax1 | ax23[3]
                return hstack(_axes_dict[other._parent], self, target=other, direction="l")
            else:
                return hstack(self, other) 
    
    def __add__(self, other):
        self._comeback()
        other._comeback()
        if other._type == "spacer":
            return other.__ror__(self) 

        elif self._parent is not None:
            if other._parent is not None: #other._type == "Brick" and other._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            return hstack(_axes_dict[self._parent], other, target=self)
        else:
            if other._parent is not None: #other._type == "Brick" and other._parent is not None: #ax1 | ax23[3]
                return hstack(_axes_dict[other._parent], self, target=other, direction="l")
            else:
                return hstack(self, other, adjust_height=False) 


    def __truediv__(self, other):
        self._comeback()
        other._comeback()
        if other._type == "spacer":
            return other.__rtruediv__(self) 

        elif other._parent is not None: #other._type == "Brick" and other._parent is not None:
            if self._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            else:
                return vstack(_axes_dict[other._parent], self, target=other)
        else:
            if self._parent is not None:
                return vstack(_axes_dict[self._parent], other, target=self, direction="b")
            else:
                return vstack(other, self)
    
    def __sub__(self, other):
        self._comeback()
        other._comeback()
        if other._type == "spacer":
            return other.__rtruediv__(self) 

        elif other._parent is not None: #other._type == "Brick" and other._parent is not None:
            if self._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            else:
                return vstack(_axes_dict[other._parent], self, target=other)
        else:
            if self._parent is not None:
                return vstack(_axes_dict[self._parent], other, target=self, direction="b")
            else:
                return vstack(other, self, adjust_width=False)

    def _repr_png_(self):
        buf = io.BytesIO()
        self.savefig(buf, "png")
        return buf.getvalue()

    def _repr_pdf_(self):
        buf = io.BytesIO()
        self.savefig(buf, "pdf")
        return buf.getvalue()

class Brick(pBrick, axes.Axes):
    def __getattribute__(self, name):
        global _axes_dict 
        if name == "case":
            x0, x1, y0, y1 = self.get_middle_corner() 
            pos = self._case.get_position() 
            px0, px1, py0, py1 = pos.x0, pos.x1, pos.y0, pos.y1
            if (_round_it(x0,4), _round_it(x1,4), _round_it(y0,4), _round_it(y1,4)) == (_round_it(px0,4), _round_it(px1,4), _round_it(py0,4), _round_it(py1,4)):
                pass 
            else:
                self._middle_flag = False
                self._outer_flag  = False
                self._case.set_position([x0, y0, x1-x0, y1-y0])
                _reset_ggplot_legend(self)
            return self._case
        
        elif name == "label":
            return self._label
        
        elif name == "outline":
            self._middle_flag = False
            self._outer_flag  = False
            x0, x1, y0, y1 = self.get_outer_corner() 
            new_dict = {} 
            for key in self.bricks_dict:
                new_dict[key] = self.bricks_dict[key] 
            
            outline_label = "outline:{}".format(self._label)
            if outline_label in pBrick._labelset:
                ax = _axes_dict[outline_label] 
            else: 
                ax = Brick(label=outline_label) 
            ax.set_position([x0, y0, x1-x0, y1-y0]) 
            ax.patch.set_facecolor("#FFFFFF") 
            ax.patch.set_alpha(0.0) 
            ax.spines["right"].set_visible(False)   
            ax.spines["top"].set_visible(False) 
            ax.spines["bottom"].set_visible(False) 
            ax.spines["left"].set_visible(False) 
            ax.set_xticks([]) 
            ax.set_yticks([])
            new_dict[outline_label] = ax
            bricks = Bricks(bricks_dict=new_dict, label="Bricks-"+outline_label)  
            bricks._case_labels = bricks._case_labels + self._case_labels
            return bricks  
        else:
            return super().__getattribute__(name) 


    def __init__(self, label=None, figsize=None, ax=None):
        """

        Parameters
        ----------
        label : str
            Unique identifier for the Bricks class object. The value can be used in layout 
            adjustment using label indexing. The value would be assigned to `self.label`.
        figsize : tuple (float, float) 
            Size of the axes (Brick) object. The unit consistent with `inch` of matplotlib.
         
        Attributes
        ----------
        case : matplotlib.Axes.axes
            Invisible axes object surrounding Brick object. 
        outline : patchworklib.Bricks
            New bricks object based on the invisible axes object surrounding all objects 
            in the original Bricks object including `case` axes.
        label : str
            Unique identifier of the Bricks class object. If the Bricks object is 
            incorporated in the other super Bricks objects, by specifying the label name 
            for the super object as `Bricks_object[{label}]`, the Bricks object can be 
            accessed from the super Bricks object.
        
        """
        global _basefigure 

        if ax is None:
            if figsize is None:
                figsize = (1,1) 
            global _axes_dict
            if "__base__" not in _axes_dict:
                ax = _basefigure.add_axes([0,0,1,1], label="__base__")
                ax.set_axis_off()
                ax.patch.set_alpha(0.0) 
                _axes_dict["__base__"] = ax 
            else:
                pass 
            axes.Axes.__init__(self, fig=_basefigure, rect=[0, 0, figsize[0], figsize[1]]) 
            _basefigure.add_axes(self) 
            if label is None:
                label = "ax_{}".format(pBrick.axnum) 
                pBrick.axnum += 1
                #raise TypeError("__init__() missing 1 required positional argument: 'label'") 
            
            if label in pBrick._labelset:
                raise ValueError("'label' value should be unique in 'Brick._labelset'")
            pBrick._labelset.add(label) 
            self.set_label(label) 
            self.bricks_dict        = {}  
            self.bricks_dict[label] = self
            _axes_dict[label]       = self
            self._label             = label 
            self._labels            = set([label])
            self._type              = "Brick"
            self._originalsize      = figsize
            self._parent = None
                    
        else:
            pos = ax.get_position()
            if figsize is None:
                figsize = (abs(pos.x1-pos.x0), abs(pos.y1-pos.y0))

            self.__dict__ = ax.__dict__ 
            if label is None:
                label = "ax_{}".format(pBrick.axnum) 
                pBrick.axnum += 1
                #raise TypeError("__init__() missing 1 required positional argument: 'label'")             
            
            if label in pBrick._labelset:
                raise ValueError("'label' value should be unique in 'Brick._labelset'")   
            
            pBrick._labelset.add(label) 
            self.set_label(label) 
            self.bricks_dict        = {}  
            self.bricks_dict[label] = self
            _axes_dict[label]       = self
            self._label             = label 
            self._labels            = set([label]) 
            self._type              = "Brick"
            self._originalsize      = figsize
            self._parent = None
        
        self._middle_flag = False
        self._outer_flag  = False
        self._case = _basefigure.add_axes([0,0,1,1], label="case:" + self._label)
        x0, x1, y0, y1 = self.get_middle_corner() 
        self._case.set_position([x0, y0, x1-x0, y1-y0])
        self._case.patch.set_facecolor("#FFFFFF") 
        self._case.patch.set_alpha(0.0) 
        self._case.spines["right"].set_visible(False)   
        self._case.spines["top"].set_visible(False) 
        self._case.spines["bottom"].set_visible(False) 
        self._case.spines["left"].set_visible(False) 
        self._case.set_xticks([]) 
        self._case.set_yticks([])
        _axes_dict[self._case.get_label()] = self._case 
        self._case_labels = [self._case.get_label()] 


class cBrick(pBrick, matplotlib.projections.polar.PolarAxes): 
    def __getattribute__(self, name):
        global _axes_dict 
        if name == "case":
            x0, x1, y0, y1 = self.get_middle_corner() 
            pos = self._case.get_position() 
            px0, px1, py0, py1 = pos.x0, pos.x1, pos.y0, pos.y1
            if (_round_it(x0,4), _round_it(x1,4), _round_it(y0,4), _round_it(y1,4)) == (_round_it(px0,4), _round_it(px1,4), _round_it(py0,4), _round_it(py1,4)):
                pass 
            else:
                self._middle_flag = False
                self._outer_flag  = False
                self._case.set_position([x0, y0, x1-x0, y1-y0])
                _reset_ggplot_legend(self)
            return self._case
        
        elif name == "label":
            return self._label
        
        elif name == "outline":
            self._middle_flag = False
            self._outer_flag  = False
            x0, x1, y0, y1 = self.get_outer_corner() 
            new_dict = {} 
            for key in self.bricks_dict:
                new_dict[key] = self.bricks_dict[key] 
            
            outline_label = "outline:{}".format(self._label)
            if outline_label in pBrick._labelset:
                ax = _axes_dict[outline_label] 
            else: 
                ax = Brick(label=outline_label) 
            ax.set_position([x0, y0, x1-x0, y1-y0]) 
            ax.patch.set_facecolor("#FFFFFF") 
            ax.patch.set_alpha(0.0) 
            ax.spines["right"].set_visible(False)   
            ax.spines["top"].set_visible(False) 
            ax.spines["bottom"].set_visible(False) 
            ax.spines["left"].set_visible(False) 
            ax.set_xticks([]) 
            ax.set_yticks([])
            new_dict[outline_label] = ax
            bricks = Bricks(bricks_dict=new_dict, label="Bricks-"+outline_label)  
            bricks._case_labels = bricks._case_labels + self._case_labels
            return bricks  
        else:
            return super().__getattribute__(name) 


    def __init__(self, label=None, figsize=None, ax=None):
        """

        Parameters
        ----------
        label : str
            Unique identifier for the Bricks class object. The value can be used in layout 
            adjustment using label indexing. The value would be assigned to `self.label`.
        figsize : tuple (float, float) 
            Size of the axes (Brick) object. The unit consistent with `inch` of matplotlib.
         
        Attributes
        ----------
        case : matplotlib.Axes.axes
            Invisible axes object surrounding Brick object. 
        outline : patchworklib.Bricks
            New bricks object based on the invisible axes object surrounding all objects 
            in the original Bricks object including `case` axes.
        label : str
            Unique identifier of the Bricks class object. If the Bricks object is 
            incorporated in the other super Bricks objects, by specifying the label name 
            for the super object as `Bricks_object[{label}]`, the Bricks object can be 
            accessed from the super Bricks object.
        
        """
        global _basefigure 

        if ax is None:
            if figsize is None:
                figsize = (1,1) 
            global _axes_dict
            if "__base__" not in _axes_dict:
                ax = _basefigure.add_axes([0,0,1,1], label="__base__")
                ax.set_axis_off()
                ax.patch.set_alpha(0.0) 
                _axes_dict["__base__"] = ax 
            else:
                pass 
            axes.Axes.__init__(self, fig=_basefigure, rect=[0, 0, figsize[0], figsize[1]]) 
            _basefigure.add_axes(self) 
            if label is None:
                label = "ax_{}".format(pBrick.axnum) 
                pBrick.axnum += 1
                #raise TypeError("__init__() missing 1 required positional argument: 'label'") 
            
            if label in pBrick._labelset:
                raise ValueError("'label' value should be unique in 'Brick._labelset'")
            pBrick._labelset.add(label) 
            self.set_label(label) 
            self.bricks_dict        = {}  
            self.bricks_dict[label] = self
            _axes_dict[label]       = self
            self._label             = label 
            self._labels            = set([label])
            self._type              = "Brick"
            self._originalsize      = figsize
            self._parent = None
                    
        else:
            pos = ax.get_position()
            if figsize is None:
                figsize = (abs(pos.x1-pos.x0), abs(pos.y1-pos.y0))

            self.__dict__ = ax.__dict__ 
            if label is None:
                label = "ax_{}".format(pBrick.axnum) 
                pBrick.axnum += 1
                #raise TypeError("__init__() missing 1 required positional argument: 'label'")             
            
            if label in pBrick._labelset:
                raise ValueError("'label' value should be unique in 'Brick._labelset'")   
            
            pBrick._labelset.add(label) 
            self.set_label(label) 
            self.bricks_dict        = {}  
            self.bricks_dict[label] = self
            _axes_dict[label]       = self
            self._label             = label 
            self._labels            = set([label]) 
            self._type              = "Brick"
            self._originalsize      = figsize
            self._parent = None
        
        self._middle_flag = False
        self._outer_flag  = False
        self._case = _basefigure.add_axes([0,0,1,1], label="case:" + self._label)
        x0, x1, y0, y1 = self.get_middle_corner() 
        self._case.set_position([x0, y0, x1-x0, y1-y0])
        self._case.patch.set_facecolor("#FFFFFF") 
        self._case.patch.set_alpha(0.0) 
        self._case.spines["right"].set_visible(False)   
        self._case.spines["top"].set_visible(False) 
        self._case.spines["bottom"].set_visible(False) 
        self._case.spines["left"].set_visible(False) 
        self._case.set_xticks([]) 
        self._case.set_yticks([])
        _axes_dict[self._case.get_label()] = self._case 
        self._case_labels = [self._case.get_label()] 

class spacer():
    def __init__(self, brick=None, value=1.0):
        self.target = brick
        self.target._middle_flag = False
        self.target._outer_flag  = False
        self.value  = value 
        self._type  = "spacer"

    def __truediv__ (self, other):
        global param
        margin = param["margin"]
        param["margin"] = None
        obj = self.resize("v") / other.outline
        param["margin"] = margin 
        return obj

    def __rtruediv__ (self, other):
        global param
        margin = param["margin"]
        param["margin"] = None
        obj = other.outline / self.resize("v")
        param["margin"] = margin 
        return obj
    
    def __or__ (self, other):
        global param
        margin = param["margin"]
        param["margin"] = None
        obj = self.resize("h") | other.outline
        param["margin"] = margin 
        return obj

    def __ror__ (self, other):
        global param
        margin = param["margin"]
        param["margin"] = None
        obj = other.outline | self.resize("h") 
        param["margin"] = margin 
        return obj
    
    def _comeback(self):  
        global _removed_axes
        global _basefigure
        fig  = _basefigure
        for label, ax in _removed_axes.items():
            ax.figure = fig
            fig.add_axes(ax)
        _removed_axes = {}

    def resize(self, direction):
        x0, x1, y0, y1 = self.target.get_outer_corner()
        if direction == "h":
            width  = abs(x1-x0) * self.value
            height = abs(y1-y0) 
            eax = Brick(figsize=(width, height))
            eax.patch.set_facecolor("#FFFFFF") 
            eax.patch.set_alpha(0.0) 
            eax.spines["right"].set_visible(False)   
            eax.spines["top"].set_visible(False) 
            eax.spines["bottom"].set_visible(False) 
            eax.spines["left"].set_visible(False) 
            eax.set_xticks([]) 
            eax.set_yticks([])
        
        if direction == "v":
            width  = abs(x1-x0) 
            height = abs(y1-y0) * self.value 
            eax = Brick(figsize=(width, height))
            eax.patch.set_facecolor("#FFFFFF") 
            eax.patch.set_alpha(0.0) 
            eax.spines["right"].set_visible(False)   
            eax.spines["top"].set_visible(False) 
            eax.spines["bottom"].set_visible(False) 
            eax.spines["left"].set_visible(False) 
            eax.set_xticks([]) 
            eax.set_yticks([])
        return eax 


if __name__ == "__main__":
    import seaborn as sns
    import numpy  as np 
    import pandas as pd 
    
    #Sample data
    fmri = sns.load_dataset("fmri")
    tips = sns.load_dataset("tips")
    diamonds = sns.load_dataset("diamonds")
    rs = np.random.RandomState(365)
    values = rs.randn(365, 4).cumsum(axis=0)
    dates = pd.date_range("1 1 2016", periods=365, freq="D")
    data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
    data = data.rolling(7).mean()
    
    #ax1
    ax1 = Brick("ax1", figsize=[3,2]) 
    sns.lineplot(x="timepoint", y="signal", hue="region", 
        style="event", data=fmri, ax=ax1)
    ax1.legend(bbox_to_anchor=(1.05, 0.8), loc='upper left')
    ax1.set_title("Brick1")

    #ax2
    ax2 = Brick("ax2", figsize=[2,4]) 
    ax2.plot([1,2,3], [1,2,3], label="line1") 
    ax2.plot([3,2,1], [1,2,3], label="line2") 
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_title("ax2")

    #Brick3
    ax3 = Brick("ax3", (4,2))
    sns.histplot(diamonds, x="price", hue="cut", multiple="stack",
        palette="light:m_r", edgecolor=".3", linewidth=.5, log_scale=True,
        ax = ax3)
    ax3.set_title("ax3")

    #Brick4
    ax4 = Brick("ax4", (6,2)) 
    sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker",
        split=True, inner="quart", linewidth=1,
        palette={"Yes": "b", "No": ".85"},
        ax=ax4)
    ax4.set_title("ax4")

    ax5 = Brick("ax5", (4,2)) 
    sns.lineplot(data=data, palette="tab10", linewidth=2.5, ax=ax5)
    ax5.set_title("ax5") 

    ax12345 = (ax1 | ax2 | ax3) / (ax4 | ax5) 
    ax12345.savefig("test1.pdf")
    
    #bricks2 = (brick2 | (brick5 / brick4)) / (brick1 | brick3) 
    ax21543 = (ax2 / ax1) | (ax5 / ax4 / ax3) 
    ax21543.savefig("test2.pdf") 
 
if StrictVersion(plotnine.__version__) >= StrictVersion("0.12"):
    overwrite_plotnine()


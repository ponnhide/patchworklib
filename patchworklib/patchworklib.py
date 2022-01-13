import os 
import io
import sys 
import copy
import dill 
import pickle
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt  
import matplotlib.axes as axes
from matplotlib.transforms import TransformedBbox, Affine2D
from matplotlib.offsetbox import AnchoredOffsetbox
from types import SimpleNamespace as NS
from contextlib import suppress
import warnings 
#warnings.simplefilter('ignore', SettingWithCopyWarning)
warnings.simplefilter('ignore')

#default setting
matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.rcParams['ps.fonttype']       = 42
matplotlib.rcParams['pdf.fonttype']      = 42
matplotlib.rcParams['font.sans-serif']   = ["Arial","Lucida Sans","DejaVu Sans","Lucida Grande","Verdana"]
matplotlib.rcParams['font.family']       = 'sans-serif'
matplotlib.rcParams['font.sans-serif']   = ["Arial","DejaVu Sans","Lucida Grande","Verdana"]
matplotlib.rcParams['font.size']         = 12.0
matplotlib.rcParams["axes.labelcolor"]   = "#000000"
matplotlib.rcParams["axes.linewidth"]    = 1.0
matplotlib.rcParams["xtick.major.width"] = 1.0
matplotlib.rcParams["xtick.minor.width"] = 0.5
matplotlib.rcParams["ytick.major.width"] = 1.0
matplotlib.rcParams["ytick.minor.width"] = 0.5
matplotlib.rcParams['xtick.major.pad']   = 4
matplotlib.rcParams['ytick.major.pad']   = 4
matplotlib.rcParams['xtick.major.size']  = 4
matplotlib.rcParams['ytick.major.size']  = 4

_axes_dict = {}
axes_dict = _axes_dict 
param = {"margin":0.5, "dpi":200}

def expand(bricks, w, h):
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
        reset_ggplot_legend(ax)
    
    for caselabel in bricks._case_labels:
        caselabel = caselabel[5:] 
        reset_ggplot_legend(_axes_dict[caselabel])
    
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
            reset_ggplot_legend(ax)
        
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
            reset_ggplot_legend(ax)

        for caselabel in bricks._case_labels:
            caselabel = caselabel[5:] 
            reset_ggplot_legend(_axes_dict[caselabel])
    
    return bricks 

def reset_ggplot_legend(bricks):
    if "_ggplot_legend" in bricks.__dict__ and bricks._ggplot_legend is not None:
        bricks._case.artists.remove(bricks._ggplot_legend)
        anchored_box = AnchoredOffsetbox(
            loc=bricks._ggplot_legend_loc,
            child=bricks._ggplot_legend_box,
            pad=0.,
            frameon=False,
            bbox_to_anchor=(bricks._ggplot_legend_x,bricks._ggplot_legend_y),
            bbox_transform = bricks.case.transAxes,
            borderpad=0.)
        anchored_box.set_zorder(90.1)
        bricks._case.add_artist(anchored_box)
        bricks._ggplot_legend = anchored_box
        bricks.case
    else:
        pass

def load_ggplot(ggplot=None, figsize=None):  
    def draw_labels(bricks, gcp):
        get_property = gcp.theme.themeables.property
        try:
            margin = get_property('axis_title_x', 'margin')
        except KeyError:
            pad_x = 5
        else:
            pad_x = margin.get_as('t', 'pt')

        try:
            margin = get_property('axis_title_y', 'margin')
        except KeyError:
            pad_y = 5
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
            xlabel = bricks.case.set_xlabel(
                labels.x, labelpad=pad_x)
            x,y = xlabel.get_position()
            xlabel.set_position([(px1+px2) / 2, y]) 

            py1 = (iy0 - oy0) / (oy1 - oy0)  
            py2 = (iy1 - oy0) / (oy1 - oy0) 
            ylabel = bricks.case.set_ylabel(
                labels.y, labelpad=pad_y)
            x,y = ylabel.get_position()
            ylabel.set_position([x, (py1+py2) / 2]) 

        else:
            xlabel = bricks.set_xlabel(
                labels.x, labelpad=pad_x)
            ylabel = bricks.set_ylabel(
                labels.y, labelpad=pad_y)
    
    def draw_legend(bricks, gori, gcp, figsize):
        get_property = gcp.theme.themeables.property
        legend_box   = gcp.guides.build(gcp)
        with suppress(KeyError):
            spacing = get_property('legend_box_spacing')
        position = gori.guides.position
        if position == 'right':
            loc = 6
            x = 1.0 + spacing/figsize[0]
            y = 0.5
        elif position == 'left':
            loc = 7
            x = 0.0 - spacing/figsize[0]
            y = 0.5
        elif position == 'top':
            loc = 8
            x = 0.5
            y = 1.0 + spacing/figsize[1]
        elif position == 'bottom':
            loc = 9
            x = 0.5
            y = 0.0- spacing/figsize[1]
        else:
            loc = 1
            x, y = position  
       
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
            bricks.case.add_artist(anchored_box)
            bricks._ggplot_legend     = anchored_box
            bricks._ggplot_legend_box = legend_box  
            bricks._ggplot_legend_loc = loc
            bricks._ggplot_legend_x   = x
            bricks._ggplot_legend_y   = y 

    def draw_title(bricks, gcp, figsize):
        title = gcp.labels.get('title', '')
        rcParams = gcp.theme.rcParams
        get_property = gcp.theme.themeables.property
        try:
            fontsize = get_property('plot_title', 'size')
        except KeyError:
            fontsize = float(rcParams.get('font.size', 12))
        try:
            margin = get_property('plot_title', 'margin')
        except KeyError:
            pad = 3
        else:
            pad = margin.get_as('b', 'in') / 0.09,
        bricks.case.set_title(title, pad=pad[0], fontsize=fontsize)
    
    #save_original_position
    global _axes_dict
    position_dict = {} 
    tmp_axes_keys = [key for key in list(_axes_dict.keys()) if type(_axes_dict[key]) == axes.Axes or _axes_dict[key]._type == "Brick"]
    for key in tmp_axes_keys:
        axtmp = _axes_dict[key] 
        position_dict[key] = axtmp.get_position() 

    gcp = copy.deepcopy(ggplot) 
    fig, gcp   = gcp.draw(return_ggplot=True)
    if figsize is None:
        figsize = fig.get_size_inches()  
    _themeable = fig._themeable
    strips     = gcp.facet.strips

    ggplot._build()
    Brick._figure._themeable = _themeable
    axs = ggplot.facet.make_axes(
        Brick._figure,
        ggplot.layout.layout,
        ggplot.coordinates) 
    
    ggplot.figure = Brick._figure
    ggplot.axs = axs
    ggplot._setup_parameters()
    ggplot.facet.strips.generate()  
    for i in range(len(ggplot.facet.strips)):
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
    ggplot._resize_panels()
    
    #Drawing
    ggplot._draw_layers()
    ggplot._draw_breaks_and_labels()
    #ggplot._draw_legend()
    #ggplot._draw_title()
    ggplot._draw_watermarks()
    ggplot._apply_theme()
    
    if len(ggplot.axs) == 1: 
        ax = Brick(ax=ggplot.axs[0])
        if "_ggplot_legend" in ax.__dict__:
            ax._ggplot_legend = None #For Google colab... 
        ax.change_aspectratio((figsize[0], figsize[1])) 
        draw_labels(ax, gcp) 
        draw_legend(ax, ggplot, gcp, figsize)
        draw_title(ax, gcp, figsize)
        plt.close(fig) 
        del gcp 
        for key in tmp_axes_keys:
            axtmp = _axes_dict[key] 
            axtmp.set_position(position_dict[key]) 
        return ax
    
    else:
        width, height = figsize 
        bricks_dict = {}
        row_list = [] 
        for ax, axcp in zip(axs, gcp.axs):
            oripos = axcp.get_position() 
            ax.set_position([oripos.x0, oripos.y0, oripos.x1-oripos.x0, oripos.y1-oripos.y0])
            brick = Brick(ax=ax) 
            bricks_dict[brick.get_label()] = brick 
            #print(brick.artists) 
            #type(brick.artists) 
        bricks = Bricks(bricks_dict=bricks_dict) 
        bricks = expand(bricks, width, height)        
        draw_labels(bricks, gcp) 
        draw_legend(bricks, ggplot, gcp, figsize)
        draw_title(bricks, gcp, figsize)
        plt.close(fig) 
        del gcp 
        for key in tmp_axes_keys:
            ax = _axes_dict[key] 
            ax.set_position(position_dict[key]) 
        return bricks

def clear():
    global _axes_dict
    _axes_dict = {}
    for ax in Brick._figure.axes:
        ax.remove() 
        del ax 
    Brick._labelset = set([]) 

def hstack(brick1, brick2, target=None, margin=None, direction="r", adjust=True):
    global param 
    global _axes_dict 
    brick1.case
    brick2.case

    ax_adjust = None
    if margin is None:
        margin = param["margin"] 

    if brick1._type == "Brick":
        brick1.set_position([0, 0, brick1._originalsize[0], brick1._originalsize[1]]) 
        brick1._parent = None  
        target = None
        labels = None
    
    if brick2._type == "Brick":
        brick2._parent = None  
        brick2.set_position([0, 0, brick2._originalsize[0], brick2._originalsize[1]]) 

    if target is not None:
        parent = brick1
        brick1_bricks_dict = brick1.bricks_dict
        if type(target) is str:
            brick1 = brick1.bricks_dict[target]
            brick1._parent = None
            labels = None

        elif type(target) is tuple:
            if type(target[0]) is str:
                labels = target
            else:
                labels = [t._label for t in target]
        
        else:
            if target._item is None:
                brick1 = target
                brick1._parent = None
                labels = None
            else:
                labels = target._item
                target._item = None
    
    else:
        brick1_bricks_dict = brick1.bricks_dict
        labels = None 

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
    
    expand(brick2, vratio, vratio) 
    if target is not None: 
        parent_icorners = parent.get_inner_corner()
        brick2_icorners = brick2.get_inner_corner() 
        if adjust == True:
            hlength = parent_icorners[1] - (margin + brick1_ocorners[1] + abs(brick2_ocorners[0]-brick2_icorners[0]))
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
                    ax_adjust = parent.bricks_dict[keys[0][1]]
                    for key in brick2.bricks_dict:
                        ax  = brick2.bricks_dict[key] 
                        pos = ax.get_position()
                        if direction == "r":
                            ax.set_position([pos.x0, pos.y0, parent_icorners[1] - ax_adjust.get_inner_corner()[0], abs(pos.y1-pos.y0)]) 
                        else:
                            ax.set_position([pos.x0, pos.y0, abs(parent_icorners[0] - ax_adjust.get_inner_corner()[1]), abs(pos.y1-pos.y0)]) 
                else:
                    adjust = False
                    hratio = hlength / abs(brick2_icorners[1] - brick2_icorners[0])  
                    for key in brick2.bricks_dict:
                        ax  = brick2.bricks_dict[key] 
                        pos = ax.get_position()
                        ax.set_position([pos.x0*hratio, pos.y0, abs(pos.x1-pos.x0) * hratio, abs(pos.y1-pos.y0)]) 
            else:
                adjust = False

    brick1_ocorners = brick1.get_outer_corner() 
    brick2_ocorners = brick2.get_outer_corner() 
    brick1_icorners = brick1.get_inner_corner()  
    brick2_icorners = brick2.get_inner_corner() 
    for key in brick2.bricks_dict:
        ax  = brick2.bricks_dict[key] 
        pos = ax.get_position()
        if ax_adjust is None:
            if margin is not None:
                if direction == "r":
                    ax.set_position([margin + brick1_ocorners[1] + abs(brick2_ocorners[0]-brick2_icorners[0]) + pos.x0 - brick2_icorners[0], pos.y0 - brick2_icorners[2] + brick1_icorners[2], pos.x1-pos.x0, pos.y1-pos.y0])
                elif direction == "l":
                    ax.set_position([abs(brick2_ocorners[0]-brick2_icorners[0]) + pos.x0 - brick2_icorners[0] + brick1_ocorners[0] - margin - (brick2_ocorners[1]-brick2_ocorners[0]), pos.y0 - brick2_icorners[2] + brick1_icorners[2], pos.x1-pos.x0, pos.y1-pos.y0])
            else:
                if direction == "r":
                    ax.set_position([brick1_icorners[1] + pos.x0 - brick2_icorners[0], pos.y0 - brick2_icorners[2] + brick1_icorners[2], pos.x1-pos.x0, pos.y1-pos.y0])
                elif direction == "l":
                    ax.set_position([pos.x0 - brick2_icorners[0] + brick1_icorners[0] - (brick2_ocorners[1]-brick2_ocorners[0]), pos.y0 - brick2_icorners[2] + brick1_icorners[2], pos.x1-pos.x0, pos.y1-pos.y0])
        else:
            if direction == "r":
                ax.set_position([ax_adjust.get_inner_corner()[0], pos.y0 - brick2_icorners[2] + brick1_icorners[2], pos.x1-pos.x0, pos.y1-pos.y0])
            elif direction == "l":
                ax.set_position([ax_adjust.get_inner_corner()[1] - (pos.x1-pos.x0), pos.y0 - brick2_icorners[2] + brick1_icorners[2], pos.x1-pos.x0, pos.y1-pos.y0])
        pos = ax.get_position()
        reset_ggplot_legend(ax)
    
    for caselabel in brick2._case_labels:
        caselabel = caselabel[5:] 
        reset_ggplot_legend(_axes_dict[caselabel])

    bricks_dict = {}
    for key in brick1_bricks_dict:
        bricks_dict[key] = brick1_bricks_dict[key] 
    
    for key in brick2.bricks_dict:
        bricks_dict[key] = brick2.bricks_dict[key]
    
    new_bricks = Bricks(bricks_dict)
    new_bricks._brick1  = _axes_dict[brick1._label]
    new_bricks._brick2  = _axes_dict[brick2._label]
    new_bricks._command = "hstack"
    if target is None:
        new_bricks._target = None
    else:
        new_bricks._target = _axes_dict[brick1._label]
    
    brick1.case
    brick2.case
    new_bricks._case_labels = new_bricks._case_labels + brick1._case_labels + brick2._case_labels
    return new_bricks

def vstack(brick1, brick2, target=None, margin=None, direction="t", adjust=True):
    global param 
    brick1.case
    brick2.case

    if margin is None:
        margin = param["margin"] 
    ax_adjust = None

    if brick1._type == "Brick":
        brick1.set_position([0, 0, brick1._originalsize[0], brick1._originalsize[1]]) 
        brick1._parent = None
        labels = None
    
    if brick2._type == "Brick":
        brick2._parent = None
        brick2.set_position([0, 0, brick2._originalsize[0], brick2._originalsize[1]]) 

    if target is not None:
        parent = brick1
        brick1_bricks_dict = brick1.bricks_dict
        if type(target) is str:
            brick1 = brick1.bricks_dict[target]
            brick1._parent = None
            labels = None

        elif type(target) is tuple:
            if type(target[0]) is str:
                labels = target
            else:
                labels = [t._label for t in target]
        else:
            if target._item is None:
                brick1 = target
                brick1._parent = None
                labels = None
            else:
                labels = target._item
                target._item = None
    else:
        brick1_bricks_dict = brick1.bricks_dict
        labels = None
    
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
    
    expand(brick2, hratio, hratio) 
    if target is not None: 
        parent_icorners = parent.get_inner_corner()
        brick2_icorners = brick2.get_inner_corner() 
        if adjust == True:
            vlength = parent_icorners[3] - (margin + brick1_ocorners[3] + abs(brick2_ocorners[2] - brick2_icorners[2])) 
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
                    ax_adjust = parent.bricks_dict[keys[0][1]]
                    for key in brick2.bricks_dict:
                        ax  = brick2.bricks_dict[key] 
                        pos = ax.get_position()
                        if direction == "t": 
                            ax.set_position([pos.x0, pos.y0,  abs(pos.x1-pos.x0), parent_icorners[3] - ax_adjust.get_inner_corner()[2]]) 
                        else:
                            ax.set_position([pos.x0, pos.y0,  abs(pos.x1-pos.x0), abs(parent_icorners[2] - ax_adjust.get_inner_corner()[3])])
                        reset_ggplot_legend(ax)
                    
                    for caselabel in brick2._case_labels:
                        caselabel = caselabel[5:] 
                        reset_ggplot_legend(_axes_dict[caselabel])
                else:
                    adjust = False
                    vratio  = vlength / abs(brick2_icorners[3] - brick2_icorners[2])  
                    for key in brick2.bricks_dict:
                        ax  = brick2.bricks_dict[key] 
                        pos = ax.get_position()
                        ax.set_position([pos.x0, pos.y0*vratio, abs(pos.x1-pos.x0), abs(pos.y1-pos.y0) * vratio])  
                        reset_ggplot_legend(ax) 
                    for caselabel in brick2._case_labels:
                        caselabel = caselabel[5:] 
                        reset_ggplot_legend(_axes_dict[caselabel])
            else:
                adjust = False

       
    brick1_ocorners = brick1.get_outer_corner() 
    brick2_ocorners = brick2.get_outer_corner() 
    brick1_icorners = brick1.get_inner_corner()  
    brick2_icorners = brick2.get_inner_corner() 
    
    for key in brick2.bricks_dict:
        ax  = brick2.bricks_dict[key] 
        pos = ax.get_position()
        if ax_adjust is None:
            if margin is not None:
                if direction == "t":
                    ax.set_position([pos.x0 - brick2_icorners[0] + brick1_icorners[0], margin + pos.y0 - brick2_icorners[2] + brick1_ocorners[3] + abs(brick2_ocorners[2] - brick2_icorners[2]) , pos.x1-pos.x0, pos.y1-pos.y0])
                elif direction == "b":
                    ax.set_position([pos.x0 - brick2_icorners[0] + brick1_icorners[0], pos.y0 - brick2_icorners[2] + abs(brick2_ocorners[2] - brick2_icorners[2]) - margin + brick1_ocorners[2] - (brick2_ocorners[3]-brick2_ocorners[2]), pos.x1-pos.x0, pos.y1-pos.y0])
            else:
                if direction == "t":
                    ax.set_position([pos.x0 - brick2_icorners[0] + brick1_icorners[0], pos.y0 - brick2_icorners[2] + brick1_icorners[3], pos.x1-pos.x0, pos.y1-pos.y0])
                elif direction == "b":
                    ax.set_position([pos.x0 - brick2_icorners[0] + brick1_icorners[0], pos.y0 - brick2_icorners[2] + brick1_icorners[2] - (brick2_ocorners[3]-brick2_ocorners[2]), pos.x1-pos.x0, pos.y1-pos.y0])
        else:
            if direction == "t":
                ax.set_position([pos.x0 - brick2_icorners[0] + brick1_icorners[0], ax_adjust.get_inner_corner()[2], pos.x1-pos.x0, pos.y1-pos.y0])
            elif direction == "b":
                ax.set_position([pos.x0 - brick2_icorners[0] + brick1_icorners[0], ax_adjust.get_inner_corner()[3]-(pos.y1-pos.y0), pos.x1-pos.x0, pos.y1-pos.y0])
        pos = ax.get_position() 
        reset_ggplot_legend(ax)

    for caselabel in brick2._case_labels:
        caselabel = caselabel[5:] 
        reset_ggplot_legend(_axes_dict[caselabel])

    bricks_dict = {}
    for key in brick1_bricks_dict:
        bricks_dict[key] = brick1_bricks_dict[key] 

    for key in brick2.bricks_dict:
        bricks_dict[key] = brick2.bricks_dict[key] 

    new_bricks = Bricks(bricks_dict) 
    new_bricks._brick1  = _axes_dict[brick1._label]
    new_bricks._brick2  = _axes_dict[brick2._label]
    new_bricks._command = "vstack"
    if target is None:
        new_bricks._target = None
    else:
        new_bricks._target = _axes_dict[brick1._label]
   
    brick1.case
    brick2.case
    new_bricks._case_labels = new_bricks._case_labels + brick1._case_labels + brick2._case_labels 
    return new_bricks

class Bricks():
    num = 0
    def __getitem__(self, item):
        if type(item) == str:
            self.bricks_dict[item]._parent = self._label
            return self.bricks_dict[item]
        
        if type(item) == tuple:
            self.bricks_dict[item[0]]._parent = self._label
            self.bricks_dict[item[0]]._item   = item
            return self.bricks_dict[item[0]] 

    def __getattribute__(self, name):
        if name == "case":
            x0, x1, y0, y1 = self.get_middle_corner() 
            pos = self._case.get_position() 
            px0, px1, py0, py1 = pos.x0, pos.x1, pos.y0, pos.y1
            if (x0, x1, y0, y1) == (px0, px1, py0, py1):
                pass 
            else:
                self._case.set_position([x0, y0, x1-x0, y1-y0])
            return self._case
        
        elif name == "outline":
            x0, x1, y0, y1 = self.get_outer_corner() 
            new_dict = {} 
            for key in self.bricks_dict:
                new_dict[key] = self.bricks_dict[key]
            outline_label = "outline:{}".format(self._label)
            if outline_label in Brick._labelset:
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

    def __init__(self, bricks_dict=None, ggplot=None, label=None): 
        global _axes_dict
        if label is None:
            self._label = "Bricks-" + str(Bricks.num)
        else:
            self._label = label
        _axes_dict[self._label] = self
        self.bricks_dict = bricks_dict 
        self._type = "Bricks"
        Bricks.num += 1
        #self._case = Brick(label="case:" + self._label) 
        self._case = Brick._figure.add_axes([0,0,1,1], label="case:" + self._label)
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
        
    def get_label(self):
        return self._label

    def get_inner_corner(self, labels=None):
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
        x0_list = [] 
        x1_list = [] 
        y0_list = [] 
        y1_list = [] 
        if labels is None:
            labels = set(self.bricks_dict.keys()) 
        for key in self.bricks_dict:
            if key in labels:
                ax   = self.bricks_dict[key]  
                h, v = Brick._figure.get_size_inches()
                pos  = ax.get_tightbbox(Brick._figure.canvas.get_renderer())
                pos  = TransformedBbox(pos, Affine2D().scale(1./Brick._figure.dpi))
                x0_list.append(pos.x0/h) 
                x1_list.append(pos.x1/h)
                y0_list.append(pos.y0/v) 
                y1_list.append(pos.y1/v) 
        return min(x0_list), max(x1_list), min(y0_list), max(y1_list) 
    
    def get_outer_corner(self):
        global _axes_dict
        x0_list = [] 
        x1_list = [] 
        y0_list = [] 
        y1_list = []
        
        for key in self.bricks_dict:
            ax   = self.bricks_dict[key]  
            h, v = Brick._figure.get_size_inches()
            pos  = ax.get_tightbbox(Brick._figure.canvas.get_renderer())
            pos  = TransformedBbox(pos, Affine2D().scale(1./Brick._figure.dpi))
            x0_list.append(pos.x0/h) 
            x1_list.append(pos.x1/h)
            y0_list.append(pos.y0/v) 
            y1_list.append(pos.y1/v)
        
        for caselabel in self._case_labels:
            h, v = Brick._figure.get_size_inches()
            pos  = _axes_dict[caselabel[5:]].case.get_tightbbox(Brick._figure.canvas.get_renderer())
            pos  = TransformedBbox(pos, Affine2D().scale(1./Brick._figure.dpi))
            x0_list.append(pos.x0/h) 
            x1_list.append(pos.x1/h)
            y0_list.append(pos.y0/v) 
            y1_list.append(pos.y1/v)  
        
        return min(x0_list), max(x1_list), min(y0_list), max(y1_list)

    def savefig(self, fname=None, transparent=None, **kwargs):
        global param
        self.case
        bytefig = io.BytesIO()  
        key0 = list(self.bricks_dict.keys())[0] 
        #pickle.dump(self.bricks_dict[key0].__class__._figure, bytefig)
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
    
    def __or__(self, other):
        if other._type == "spacer":
            return other.__ror__(self) 
        elif other._type == "Brick" and other._parent is not None:
            return hstack(_axes_dict[other._parent], self, target=other, direction="l")
        else:
            return hstack(self, other)
    
    def __truediv__(self, other):
        if other._type == "spacer":
            return other.__rtruediv__(self) 
        elif other._type == "Brick" and other._parent is not None:
            return vstack(_axes_dict[other._parent], self, target=other)
        else:
            return vstack(other, self)

class Brick(axes.Axes): 
    axnum = 0    
    _figure   = plt.figure(figsize=(1,1))   
    _labelset = set([]) 
    
    def __getattribute__(self, name):
        global _axes_dict 
        if name == "case":
            x0, x1, y0, y1 = self.get_middle_corner() 
            pos = self._case.get_position() 
            px0, px1, py0, py1 = pos.x0, pos.x1, pos.y0, pos.y1
            if (x0, x1, y0, y1) == (px0, px1, py0, py1):
                pass 
            self._case.set_position([x0, y0, x1-x0, y1-y0])
            return self._case
        
        elif name == "outline":
            x0, x1, y0, y1 = self.get_outer_corner() 
            new_dict = {} 
            for key in self.bricks_dict:
                new_dict[key] = self.bricks_dict[key] 
            #for key in self._case_labels: 
            #    new_dict[key] = _axes_dict[key]
            outline_label = "outline:{}".format(self._label)
            if outline_label in Brick._labelset:
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

    def __init__(self, label=None, figsize=(1,1), ax=None):
        if ax is None:
            global _axes_dict
            if "__base__" not in _axes_dict:
                ax = Brick._figure.add_axes([0,0,1,1], label="__base__")
                ax.set_axis_off()
                ax.patch.set_alpha(0.0) 
                _axes_dict["__base__"] = ax 
            else:
                pass 
            axes.Axes.__init__(self, fig=Brick._figure, rect=[0, 0, figsize[0], figsize[1]]) 
            Brick._figure.add_axes(self) 
            if label is None:
                label = "ax_{}".format(Brick.axnum) 
                Brick.axnum += 1
                #raise TypeError("__init__() missing 1 required positional argument: 'label'") 
            
            if label in Brick._labelset:
                raise ValueError("'label' value should be unique in 'Brick._labelset'")
            Brick._labelset.add(label) 
            self.set_label(label) 
            self.bricks_dict        = {}  
            self.bricks_dict[label] = self
            _axes_dict[label]       = self
            self._label             = label 
            self._type              = "Brick"
            self._originalsize      = figsize
            self._parent = None
            self._item = None
                    
        else:
            self.__dict__ = ax.__dict__
            if label is None:
                label = "ax_{}".format(Brick.axnum) 
                Brick.axnum += 1
                #raise TypeError("__init__() missing 1 required positional argument: 'label'")             
            if label in Brick._labelset:
                raise ValueError("'label' value should be unique in 'Brick._labelset'")   
            Brick._labelset.add(label) 
            self.set_label(label) 
            self.bricks_dict        = {}  
            self.bricks_dict[label] = self
            _axes_dict[label]       = self
            self._label             = label 
            self._type              = "Brick"
            self._originalsize      = figsize
            self._parent = None
            self._item = None
        
        #self._case = Brick(label="case:" + self._label) #._figure.add_axes([0,0,1,1], label="case:" + self._label)
        self._case = Brick._figure.add_axes([0,0,1,1], label="case:" + self._label)
        x0, x1, y0, y1 = self.get_middle_corner() 
        self._case.set_position([x1, y0, x1-x0, y1-y0])
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

    def get_inner_corner(self, labels=None):
        pos = self.get_position()  
        return pos.x0, pos.x1, pos.y0, pos.y1

    def get_middle_corner(self, labels=None):
        h, v = Brick._figure.get_size_inches()
        pos  = self.get_tightbbox(Brick._figure.canvas.get_renderer())
        pos  = TransformedBbox(pos, Affine2D().scale(1./Brick._figure.dpi))
        return pos.x0/h, pos.x1/h, pos.y0/v, pos.y1/v
    
    def get_outer_corner(self, labes=None): 
        h, v = Brick._figure.get_size_inches()
        pos1  = self.get_tightbbox(Brick._figure.canvas.get_renderer())
        pos1  = TransformedBbox(pos1, Affine2D().scale(1./Brick._figure.dpi))

        pos2  = self.case.get_tightbbox(Brick._figure.canvas.get_renderer())
        pos2  = TransformedBbox(pos2, Affine2D().scale(1./Brick._figure.dpi))
        return min([pos1.x0/h, pos2.x0/h]), max([pos1.x1/h, pos2.x1/h]), min([pos1.y0/v, pos2.y0/v]), max([pos1.y1/v, pos2.y1/v])
    
    def savefig(self, fname=None, transparent=None, **kwargs):
        global param
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
   
    def change_aspectratio(self, new_size): 
        if type(new_size) ==  tuple or type(new_size) == list:
            self.set_position([0, 0, new_size[0], new_size[1]])
            self._originalsize = new_size 
        else:
            self.set_position([0, 0, 1, new_size])
            self._originalsize = (1, new_size) 
        reset_ggplot_legend(self)

    def move_legend(self, new_loc, **kws):
        old_legend = self.legend_
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        self.legend(handles, labels, loc=new_loc, title=title, **kws)
    
    def __or__(self, other):
        if other._type == "spacer":
            return other.__ror__(self) 

        elif self._parent is not None:
            if other._type == "Brick" and other._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            return hstack(_axes_dict[self._parent], other, target=self)
        else:
            if other._type == "Brick" and other._parent is not None: #ax1 | ax23[3]
                return hstack(_axes_dict[other._parent], self, target=other, direction="l")
            else:
                return hstack(self, other) 

    def __truediv__(self, other):
        if other._type == "spacer":
            return other.__rtruediv__(self) 

        elif other._type == "Brick" and other._parent is not None:
            if self._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            else:
                return vstack(_axes_dict[other._parent], self, target=other)
        else:
            if self._parent is not None:
                return vstack(_axes_dict[self._parent], other, target=self, direction="b")
            else:
                return vstack(other, self)
    
class spacer():
    def __init__(self, brick=None, value=1.0):
        self.target = brick
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
    

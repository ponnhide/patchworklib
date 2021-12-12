import os 
import io
import sys 
import copy
import pickle
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt  
import matplotlib.axes as axes
from matplotlib.transforms import TransformedBbox, Affine2D

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
param = {"margin":0.5} 
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

    brick1_ocorners = brick1.get_outer_corner(labels) 
    brick2_ocorners = brick2.get_outer_corner() 
    brick1_icorners = brick1.get_inner_corner(labels)  
    brick2_icorners = brick2.get_inner_corner() 
        
    vratio = abs(brick1_icorners[3] - brick1_icorners[2]) / abs(brick2_icorners[3] - brick2_icorners[2])  
    if vratio < 0.8 and target is None: 
        for key in brick1.bricks_dict:
            ax  = brick1.bricks_dict[key] 
            pos = ax.get_position()  
            ax.set_position([pos.x0 * 1/vratio, pos.y0 * 1/vratio, (pos.x1 - pos.x0) * 1/vratio, (pos.y1 - pos.y0) * 1/vratio]) 
        brick1_ocorners = brick1.get_outer_corner() 
        brick2_ocorners = brick2.get_outer_corner() 
        brick1_icorners = brick1.get_inner_corner()  
        brick2_icorners = brick2.get_inner_corner() 
        vratio = abs(brick1_icorners[3] - brick1_icorners[2]) / abs(brick2_icorners[3] - brick2_icorners[2])  
    
    for key in brick2.bricks_dict:
        ax  = brick2.bricks_dict[key] 
        pos = ax.get_position()
        ax.set_position([pos.x0 * vratio, pos.y0 * vratio, abs(pos.x1-pos.x0) * vratio, abs(pos.y1-pos.y0) * vratio]) 
    
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
        #print(key)
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
    return new_bricks

def vstack(brick1, brick2, target=None, margin=None, direction="t", adjust=True):
    global param 
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

    brick1_ocorners = brick1.get_outer_corner(labels) 
    brick2_ocorners = brick2.get_outer_corner() 
    brick1_icorners = brick1.get_inner_corner(labels)  
    brick2_icorners = brick2.get_inner_corner() 
    hratio = abs(brick1_icorners[1] - brick1_icorners[0]) / abs(brick2_icorners[1] - brick2_icorners[0])  
    if hratio < 1.0 and target is None: 
        for key in brick1.bricks_dict:
            ax  = brick1.bricks_dict[key] 
            pos = ax.get_position()  
            ax.set_position([pos.x0* 1/hratio, pos.y0* 1/hratio, (pos.x1 - pos.x0) * 1/hratio, (pos.y1 - pos.y0) * 1/hratio]) 
        brick1_ocorners = brick1.get_outer_corner() 
        brick2_ocorners = brick2.get_outer_corner() 
        brick1_icorners = brick1.get_inner_corner()  
        brick2_icorners = brick2.get_inner_corner() 
        hratio = abs(brick1_icorners[1] - brick1_icorners[0]) / abs(brick2_icorners[1] - brick2_icorners[0])  

    for key in brick2.bricks_dict:
        ax  = brick2.bricks_dict[key] 
        pos = ax.get_position()
        ax.set_position([pos.x0*hratio, pos.y0*hratio, abs(pos.x1-pos.x0) * hratio, abs(pos.y1-pos.y0) * hratio]) 

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
                else:
                    adjust = False
                    vratio  = vlength / abs(brick2_icorners[3] - brick2_icorners[2])  
                    for key in brick2.bricks_dict:
                        ax  = brick2.bricks_dict[key] 
                        pos = ax.get_position()
                        ax.set_position([pos.x0, pos.y0*vratio, abs(pos.x1-pos.x0), abs(pos.y1-pos.y0) * vratio])  
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

    def __init__(self, bricks_dict=None): 
        global _axes_dict
        self._label = "bricks" + str(Bricks.num)
        _axes_dict[self._label] = self
        self.bricks_dict = bricks_dict 
        self._type = "Bricks"
        Bricks.num += 1

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

    def get_outer_corner(self, labels=None): 
        x0_list = [] 
        x1_list = [] 
        y0_list = [] 
        y1_list = [] 
        if labels is None:
            labels = set(self.bricks_dict.keys()) 
        for key in self.bricks_dict:
            if key in labels:
                ax   = self.bricks_dict[key]  
                h, v = ax.__class__._figure.get_size_inches()
                pos  = ax.get_tightbbox(ax.__class__._figure.canvas.get_renderer())
                pos  = TransformedBbox(pos, Affine2D().scale(1./ax.__class__._figure.dpi))
                x0_list.append(pos.x0/h) 
                x1_list.append(pos.x1/h)
                y0_list.append(pos.y0/v) 
                y1_list.append(pos.y1/v)
        return min(x0_list), max(x1_list), min(y0_list), max(y1_list), 
 
    def savefig(self, fname=None):
        bytefig = io.BytesIO()  
        key0 = list(self.bricks_dict.keys())[0] 
        pickle.dump(self.bricks_dict[key0].__class__._figure, bytefig)
        bytefig.seek(0) 
        tmpfig = pickle.load(bytefig) 
        for ax in tmpfig.axes:
            if ax.get_label() in self.bricks_dict:
                pass 
            else:
                ax.remove() 
        
        if fname is not None:  
            tmpfig.savefig(fname, bbox_inches="tight") 
        return tmpfig 
    
    def __or__(self, other):
        if other._type == "Brick" and other._parent is not None:
            return hstack(_axes_dict[other._parent], self, target=other, direction="l")
        else:
            return hstack(self, other)
    
    def __truediv__(self, other):
        if other._type == "Brick" and other._parent is not None:
            return vstack(_axes_dict[other._parent], self, target=other)
        else:
            return vstack(other, self)

class Brick(axes.Axes): 
    axnum = 0    
    _figure   = plt.figure(figsize=(1,1))   
    _labelset = set([]) 
    def __init__(self, label=None, figsize=(1,1)):
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
        self._lael              = label 
        self._type              = "Brick"
        self._originalsize      = figsize
        self._parent = None
        self._item = None

    def get_inner_corner(self, labels=None):
        pos = self.get_position()  
        return pos.x0, pos.x1, pos.y0, pos.y1

    def get_outer_corner(self, labes=None): 
        h, v = Brick._figure.get_size_inches()
        pos  = self.get_tightbbox(Brick._figure.canvas.get_renderer())
        pos  = TransformedBbox(pos, Affine2D().scale(1./Brick._figure.dpi))
        return pos.x0/h, pos.x1/h, pos.y0/v, pos.y1/v
    
    def savefig(self, fname=None):
        bytefig = io.BytesIO()  
        pickle.dump(Brick._figure, bytefig)
        bytefig.seek(0) 
        tmpfig = pickle.load(bytefig) 
        for ax in tmpfig.axes:
            if ax.get_label() in self.bricks_dict:
                pass 
            else:
                ax.remove() 
        if fname is not None:  
            tmpfig.savefig(fname, bbox_inches="tight") 
        return tmpfig 
    
    def change_aspectratio(self, new_size): 
        if type(new_size) ==  tuple or type(new_size) == list:
            self.set_position([0, 0, new_size[0], new_size[1]])
            self._originalsize = new_size 
        else:
            self.set_position([0, 0, 1, new_size])
            self._originalsize = (1, new_size) 
    
    def move_legend(self, new_loc, **kws):
        old_legend = self.legend_
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        self.legend(handles, labels, loc=new_loc, title=title, **kws)

    def __or__(self, other):
        if self._parent is not None:
            if other._type == "Brick" and other._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            return hstack(_axes_dict[self._parent], other, target=self)
        else:
            if other._type == "Brick" and other._parent is not None:
                return hstack(_axes_dict[other._parent], self, target=other, direction="l")
            else:
                return hstack(self, other)

    def __truediv__(self, other):
        if other._type == "Brick" and other._parent is not None:
            if self._parent is not None:
                raise ValueError("Specifications of multiple targets are not supported") 
            else:
                return vstack(_axes_dict[other._parent], self, target=other)
        else:
            if self._parent is not None:
                return vstack(_axes_dict[self._parent], other, target=self, direction="b")
            else:
                return vstack(other, self)
    
   
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
    

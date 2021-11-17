# patchworklib
Matplotlib subplot functions are not optimized for interactive programming environments such as Jupyter-lab. This point seems to be discussed in various places, and the matplotlib side starts to provide a new function for quickly placing multiple graphs.

- https://twitter.com/kasparmartens/status/1381991445381406722?s=20 
- https://github.com/has2k1/plotnine/issues/46	
- [subplot_mosaic]( https://matplotlib.org/stable/tutorials/provisional/mosaic.html#sphx-glr-tutorials-provisional-mosaic-py)  

However, they probably do not understand our hope. We do not want to think of a layout of multiple graphs before drawing them. After visualizing each graph, we want to test multiple layouts for them and find the best layout. Here, I tried to implement the [patchwork](https://github.com/thomasp85/patchwork)-like module on matplotlib. You can quickly design a tidy layout for multiple graphs.

## Installation
1. Download the patchworklib package from the GitHub repository.   
    `git clone https://github.com/ponnhide/patchworklib.git`

2. Move to the patchworklib directory and install patchworklib using the following command.  
    `python setup.py install` 

### Demonstration
Jupyter Notebook files for all of the example codes are provided in ./tutorial and also made executable in [Google Colaboratory](https://colab.research.google.com/drive/1TVcH3IJy6geDXVJDfOKCPFPsP2GzjxHu?usp=sharing)

### test code
**Drawing the individual graphs**
    
```python
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
```
    
**Patchwork demo**
```python
ax12345 = (ax1 | ax2 | ax3) / (ax4 | ax5) 
ax12345.savefig("test1.pdf")

ax21543 = (ax2 / ax1) | (ax5 / ax4 / ax3) 
ax21543.savefig("test2.pdf") 
```
    
### Result
**test1**

<img src="test1.png" width="600x600">

**test2**  

<img src="test2.png" width="600x600">


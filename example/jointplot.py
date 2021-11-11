import os 
import numpy as np 
from patchworklib import patchwork as pw
pw.param["margin"] = 0.05

ax1 = pw.Brick("ax1", aspect=(3,3))  
ax2 = pw.Brick("ax2", aspect=(1,3))  
ax3 = pw.Brick("ax3", aspect=(3,1)) 

data = np.random.randn(2000,2000)
ax1.scatter(data.T[0], data.T[1], s=5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.hist(data.T[1], range=(-2,2), bins=100, orientation="horizontal") 
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_xticks([]) 
ax2.set_yticks([])

ax3.hist(data.T[1], range=(-2,2), bins=100)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False) 
ax3.set_xticks([])
ax3.set_yticks([])

ax1.set_xlim(-2,2)
ax1.set_ylim(-2,2)
ax2.set_ylim(-2,2) 
ax3.set_xlim(-2,2)

ax12  = ax1 | ax2
ax123 = ax3 / ax12["ax1"] 
ax123.savefig("joint.pdf")

ax4 = pw.Brick("ax4", aspect=(3,3))  
ax5 = pw.Brick("ax5", aspect=(1,3))  
ax6 = pw.Brick("ax6", aspect=(3,1))

data = np.random.normal(0,0.5,(2000,2000))
ax4.scatter(data.T[0], data.T[1], s=5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

ax5.hist(data.T[1], range=(-2,2), bins=100, orientation="horizontal") 
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.set_xticks([]) 
ax5.set_yticks([])

ax6.hist(data.T[1], range=(-2,2), bins=100)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_visible(False) 
ax6.set_xticks([])
ax6.set_yticks([])

ax4.set_xlim(-2,2)
ax4.set_ylim(-2,2)
ax5.set_ylim(-2,2)
ax6.set_xlim(-2,2)

ax4.set_ylabel("Patchwork demo") 

ax46  = ax6 / ax4
ax456 = ax46["ax4"] | ax5

pw.param["margin"] = 0.0
ax123456 = ax123 | ax456
ax123456.savefig("joint2.pdf") 

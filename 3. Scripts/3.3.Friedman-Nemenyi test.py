""" This code contains the code for the Friedman-Nemenyi test. """
# Import libraries
from scipy import stats
import scikit_posthocs as sp
import numpy as np
import matplotlib.pyplot as plt

""" R2 score """
print("------------------------- R2 score -------------------------")
# R2 scores for the 3 methods
qbc_r2 = [3.760, 8.857, 5.473, 0.534, 9.701, 3.151, 10.542, 11.554, 12.477, -7.105]
greedy_r2 = [3.739, 8.561, 5.227, 0.539, 9.524, 3.202, 10.504, 12.265, 12.453, -8.440]
random_r2 = [3.362, 7.815, 4.861, 0.500, 9.833, 3.004, 9.924, 12.923, 12.327, -11.912]

# Conduct the Friedman test
output = stats.friedmanchisquare(qbc_r2, greedy_r2, random_r2)

print("Methods are significantly different?" )
if output.pvalue < 0.05:
    print("Answer: Yes" )
else:
    print("Answer: No")

# Conduct the Nemenyi test
data = np.array([qbc_r2, greedy_r2, random_r2])
 
print(sp.posthoc_nemenyi_friedman(data.T))

# Plot the results
# Average ranking
cd = 1.05
qbc = 1.5
greedy = 1.9
random = 2.6

limits=(3,1)

fig, ax = plt.subplots(figsize=(5,1.8))
plt.subplots_adjust(left=0.2, right=0.8)

# set up plot
ax.set_xlim(limits)
ax.set_ylim(0,1)
ax.spines['top'].set_position(('axes', 0.6))
#ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_visible(False)
for pos in ["bottom", "left", "right"]:
    ax.spines[pos].set_visible(False)

# CD bar
ax.plot([limits[0],limits[0]-cd], [.9,.9], color="k")
ax.plot([limits[0],limits[0]], [.9-0.03,.9+0.03], color="k")
ax.plot([limits[0]-cd,limits[0]-cd], [.9-0.03,.9+0.03], color="k") 
ax.text(limits[0]-cd/2., 0.92, "CD", ha="center", va="bottom") 

# annotations
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90")
kw = dict(xycoords='data',textcoords="axes fraction",
          arrowprops=arrowprops, bbox=bbox_props, va="center")
ax.annotate("Random", xy=(random, 0.6), xytext=(0,0.25),ha="right",  **kw)
ax.annotate("QBC-RF", xy=(qbc, 0.6), xytext=(1.,0.25),ha="left",  **kw)
ax.annotate("Greedy", xy=(greedy, 0.6), xytext=(1.,0),ha="left",  **kw)

#bars
ax.plot([qbc,greedy],[0.55,0.55], color="k", lw=3)
ax.plot([greedy, random],[0.48,0.48], color="k", lw=3)

plt.savefig("..\\4.Results\\friedman_nemenyi_R2")
plt.show()

""" MAE """
print("------------------------- MAE -------------------------")
# MAE for the 3 methods
qbc_mae = [0.171, 0.074, 0.579, 0.513, 0.159, 0.207, 0.167, 0.002, 0.006, 0.281]
greedy_mae = [0.179, 0.077, 0.620, 0.512, 0.159, 0.205, 0.168, 0.002, 0.006, 0.315]
random_mae = [0.178, 0.079, 0.624, 0.513, 0.151, 0.207, 0.174, 0.002, 0.006, 0.321]

# Conduct the Friedman test
output = stats.friedmanchisquare(qbc_mae, greedy_mae, random_mae)

print("Methods are significantly different?" )
if output.pvalue < 0.05:
    print("Answer: Yes" )
else:
    print("Answer: No")

# Conduct the Nemenyi test
data = np.array([qbc_mae, greedy_mae, random_mae])
 
print(sp.posthoc_nemenyi_friedman(data.T))

# Plot the results
# Average ranking
cd = 1.05
qbc = 1.3
greedy = 1.7
random = 2.1

limits=(3,1)

fig, ax = plt.subplots(figsize=(5,1.8))
plt.subplots_adjust(left=0.2, right=0.8)

# set up plot
ax.set_xlim(limits)
ax.set_ylim(0,1)
ax.spines['top'].set_position(('axes', 0.6))
#ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_visible(False)
for pos in ["bottom", "left", "right"]:
    ax.spines[pos].set_visible(False)

# CD bar
ax.plot([limits[0],limits[0]-cd], [.9,.9], color="k")
ax.plot([limits[0],limits[0]], [.9-0.03,.9+0.03], color="k")
ax.plot([limits[0]-cd,limits[0]-cd], [.9-0.03,.9+0.03], color="k") 
ax.text(limits[0]-cd/2., 0.92, "CD", ha="center", va="bottom") 

# annotations
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90")
kw = dict(xycoords='data',textcoords="axes fraction",
          arrowprops=arrowprops, bbox=bbox_props, va="center")
ax.annotate("Random", xy=(random, 0.6), xytext=(0,0.25),ha="right",  **kw)
ax.annotate("QBC-RF", xy=(qbc, 0.6), xytext=(1.,0.25),ha="left",  **kw)
ax.annotate("Greedy", xy=(greedy, 0.6), xytext=(1.,0),ha="left",  **kw)

#bars
ax.plot([qbc,greedy],[0.55,0.55], color="k", lw=3)
ax.plot([greedy, random],[0.55,0.55], color="k", lw=3)

plt.savefig("..\\4.Results\\friedman_nemenyi_MAE")
plt.show()

""" MSE """
print("------------------------- MSE -------------------------")
# MAE for the 3 methods
qbc_mse = [0.00562, 0.00144, 0.05186, 0.02666, 0.00468, 0.00508, 0.00692, 0.00000, 0.00000, 0.04320]
greedy_mse = [0.00534, 0.00146, 0.05352, 0.02678, 0.00478, 0.00500, 0.00690, 0.00000, 0.00000, 0.04600]
random_mse = [0.00562, 0.00164, 0.05404, 0.02686, 0.00434, 0.00516, 0.00860, 0.00000, 0.00000, 0.04792]

# Conduct the Friedman test
output = stats.friedmanchisquare(qbc_mse, greedy_mse, random_mse)
print(output.pvalue)
print("Methods are significantly different?" )
if output.pvalue < 0.05:
    print("Answer: Yes" )
else:
    print("Answer: No")

# Conduct the Nemenyi test
data = np.array([qbc_mse, greedy_mse, random_mse])
 
print(sp.posthoc_nemenyi_friedman(data.T))

# Plot the results
# Average ranking
cd = 1.05
qbc = 1.4
greedy = 1.6
random = 2.3

limits=(3,1)

fig, ax = plt.subplots(figsize=(5,1.8))
plt.subplots_adjust(left=0.2, right=0.8)

# set up plot
ax.set_xlim(limits)
ax.set_ylim(0,1)
ax.spines['top'].set_position(('axes', 0.6))
#ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_visible(False)
for pos in ["bottom", "left", "right"]:
    ax.spines[pos].set_visible(False)

# CD bar
ax.plot([limits[0],limits[0]-cd], [.9,.9], color="k")
ax.plot([limits[0],limits[0]], [.9-0.03,.9+0.03], color="k")
ax.plot([limits[0]-cd,limits[0]-cd], [.9-0.03,.9+0.03], color="k") 
ax.text(limits[0]-cd/2., 0.92, "CD", ha="center", va="bottom") 

# annotations
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90")
kw = dict(xycoords='data',textcoords="axes fraction",
          arrowprops=arrowprops, bbox=bbox_props, va="center")
ax.annotate("Random", xy=(random, 0.6), xytext=(0,0.25),ha="right",  **kw)
ax.annotate("QBC-RF", xy=(qbc, 0.6), xytext=(1.,0.25),ha="left",  **kw)
ax.annotate("Greedy", xy=(greedy, 0.6), xytext=(1.,0),ha="left",  **kw)

#bars
ax.plot([qbc,greedy],[0.55,0.55], color="k", lw=3)
ax.plot([greedy, random],[0.55,0.55], color="k", lw=3)

plt.savefig("..\\4.Results\\friedman_nemenyi_MSE")
plt.show()

import numpy as np


# Problems that needs to be adressed:
# 1. Size that fits
# 2. Resolution
# 3. Put legend in a place where there is no overlap?
# 4.



# Idea for this script is to spend finally less time with Interspeech figures in Python
# Two types of diagrams are feasible in my opinion:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# 80 mm x 80 mm
plt.figure(num=None, figsize=(3.14,3.14))

x = np.arange(100)
y = np.arange(100)
plt.plot(x,y,linewidth="2")

plt.xticks()
plt.tick_params(axis='both', which='major', pad=1)
plt.tight_layout()
plt.show()
# 200 mm x 80 mm

plt.figure(num=None, figsize=(7.87,3.14))

x = np.arange(100)
y = np.arange(100)
plt.plot(x,y,linewidth="2")
plt.tight_layout()
plt.show()
"""
A set of formulas to cross analyse the ChemHammer performance
"""
import os
from random import shuffle
from copy import deepcopy

from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

from ChemHammer import ChemHammer
from MatrixSort import DistanceMatrixSorter

formula_list = ["Na0.99Zr2P3O12",     # Some NASICONs first repeated to ensure = 0
                "NaZr2P3O12",
                "Na2Zr2Si1P2O12",
                "Na3Zr2Si2P1O12",
                "Na4Zr2Si3O12",
                "LiZr 2 (PO 4 ) 3",              # Some Li NASICONs
                "Li 1.3 Al 0.3 Ge 1.7 (PO 4 ) 3",
                "Li 1.3 Al 0.3 Ti 1.7 (PO 4 ) 3",
                "Li 1.3 Fe 0.3 Hf 1.7 (PO 4 ) 3",
                "Li 0.5 Cr 0.5 Ti 1.5 (PO 4 ) 3",
                "Li Ti 2 (PO 4 ) 3",              # Two LATPs
                "Li3 Ti 2 (PO 4 ) 3",
                "Li 1.5 Al 0.5 Ge 1.5 (PO 4 ) 3", # LAGP
                "Li 1.5 Al 0.4 Cr 0.1 Ge 1.5 (PO 4 ) 3", # Al/Cr doped LAGP
                "Li 14 Zn(GeO 4 ) 4",       #LISICONs
                "Li 3.5 Ge 0.5 VO 4",
                "Li 3.5 Si 0.5 P 0.5 O 4",
                "Li 4 Al 0.3333 Si 0.1666 Ge 0.1666 P 0.3333 O 4",
                "Li 2 GeS 3" ,   # Sulfide type electrolytes
                "Li 2 ZnGeS 4",
                "Li 5 GaS 4",
                "Li 3.25 Ge 0.25 P 0.75 S 4", # Repeated compund below
                "Li 10 GeP 2 S 12",
                "Li 9.54 Si 1.74 P 1.44 S 11.7 Cl 0.3",
                "Li 3 Y 3 Te 2 O 12",    # Garnets
                "Li 3 Y 3 W 2 O 12",
                "Li 5 La 3 Nb 2 O 12",
                "Li 5 La 3 Ta 2 O 12",
                "Li 6 BaLa 2 Ta 2 O 12",
                "Li 6 SrLa 2 Ta 2 O 12",
                "Li 7 La 3 Zr 2 O 12", # LLZO Garnet
                "Li 6.75 La 3 (Zr 1.75 Nb 0.25 )O 12",
                "Li 7.06 La 3 Y 0.06 Zr 1.94 O 12",
                "Li 6.4 Ga 0.2 La 3 Zr 2 O 12",
                "Li 6.20 Ga 0.30 La 2.95 Rb 0.05 Zr 2 O 12",
                "Li 0.34La0.51TiO2.94", # Perovskites
                "Li 0.35 La 0.55 TiO 3",
                "LiSr 1.65 Zr 1.3 Ta 1.7 O 9",
                "Li0.375 Sr 0.4375 Ta 0.75 Zr0.25 O 3",
                "LiSr 1.65 Zr 1.3 Ta 1.7 O 9",
                "Li0.375 Sr0.4375Ta0.75 Hf0.25 O 3",
                "Li 3 OCl",             # anti perovskites
                "Li 3 OCl 0.5 Br 0.5"]

lev_dist = []

x = ChemHammer('Li 1.3 Al 0.3 Ge 1.7 (PO 4 ) 3')
y = ChemHammer('Li 1.3 B 0.3 Ge 1.7 (PO 4 ) 3')

print(x.formula)
print(x.normed_composition)
print(y.formula)
print(y.normed_composition)
print(x.min_flow_dist(y))


for i in range(len(formula_list[:-1])):
    x = ChemHammer(formula_list[i])
    for j in range(i+1, len(formula_list)):
        print("################")
        print(x.formula)
        print(x.normed_composition)
        y = ChemHammer(formula_list[j])
        print(y.formula)
        print(y.normed_composition)
        lev_dist.append(x.min_flow_dist(y))

lev_dist = squareform(lev_dist)

plt.subplot(131, aspect='equal')
plt.imshow(lev_dist, cmap='binary')

# Now lets shuffle the list and repeat the same as above to show disorder
shuff_lev_dist = []
shuff_list = deepcopy(formula_list)
shuffle(shuff_list)

for i in range(len(shuff_list) - 1):
    x = ChemHammer(shuff_list[i], metric="manhattan")
    for j in range(i + 1, len(formula_list)):
        shuff_lev_dist.append(x.min_flow_dist(shuff_list[j]))

shuff_lev_dist = squareform(shuff_lev_dist)

plt.subplot(132, aspect='equal')
plt.imshow(shuff_lev_dist, cmap='binary')

# "ward", "single", "average", "complete"
# Now lets resort the list and see how this comes out

sorted_mat_ward = DistanceMatrixSorter(lev_dist, method="ward")
sorted_mat_sing = DistanceMatrixSorter(lev_dist, method="single")
sorted_mat_ave = DistanceMatrixSorter(lev_dist, method="average")
sorted_mat_comp = DistanceMatrixSorter(lev_dist, method="complete")

plt.subplot(133, aspect='equal')
plt.imshow(sorted_mat_ward.ordered_dist_mat, cmap='binary')

# plt.subplot(246, aspect='equal')
# plt.pcolormesh(sorted_mat_ward.ordered_dist_mat, cmap='binary')

# plt.subplot(247, aspect='equal')
# plt.pcolormesh(sorted_mat_ward.ordered_dist_mat, cmap='binary')

# plt.subplot(248, aspect='equal')
# plt.pcolormesh(sorted_mat_ward.ordered_dist_mat, cmap='binary')

plt.show()
indices = sorted_mat_ward.sorted_index
resorted_list = [shuff_list[i-1] for i in indices]

print(resorted_list)

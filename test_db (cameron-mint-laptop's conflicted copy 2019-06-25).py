import os
import sqlite3
from time import time

import pandas as pd
import numpy as np

from pandas import Series
from pandas import ExcelWriter

from mpi4py import MPI

from ChemHammer import ChemHammer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = 0

def split_indices(input_array):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # If using MPI
    num_processes = comm.size

    if rank == 0:                # First rank splits the files up
        indexes = np.arange(len(input_array))
        np.random.shuffle(indexes)
        splits = np.array_split(indexes, num_processes)
    else:                      # All other processes
        splits = []

    # wait for process 0 to get the filepaths/splits and broadcast these
    comm.Barrier()
    splits = comm.bcast(splits, root=0)
    my_indexes = splits[comm.rank]
    # take only filepaths for our rank
    my_array = [input_array[x] for x in my_indexes]
    return my_array, my_indexes

time_start = time()

ionic_db_dir = "/home/cameron/Dropbox/University/PhD/ionic_db_labelled_ICSD.xlsx"
cif_db_dir = "/home/cameron/Dropbox/University/PhD/cif.db"
db = pd.read_excel(ionic_db_dir)
conn = sqlite3.connect(cif_db_dir)
cursor = conn.cursor()

# take all li compounds
sql_query = 'select formula, icsd_code from data where formula like "%Li%"'
cursor.execute(f'{sql_query}')
result = cursor.fetchall()

# # Add a new column for predicted ICSD codes
compounds = db['Compound Formula']
my_compounds, my_splits = split_indices(compounds)

# checked = db['Confirmed Y/N']
# my_checked = checked[my_splits]

# strings = []

print("Entering main loop")
tested_indices = []

test_crystals = [
    "Li4Al0.3333Si0.1666Ge0.1666P0.3333O4",
    "Li2GeS3", #Sulfidetypeelectrolytes
    "Li2ZnGeS4",
    "Li5GaS4",
    "Li3.25Ge0.25P0.75S4", #Repeatedcompundbelow
    "Li10GeP2S12",
    "Li9.54Si1.74P1.44S11.7Cl0.3",
    "Li3Y3Te2O12",#Garnets
    "Li3Y3W2O12",
    "Li5La3Nb2O12",
    "Li5La3Ta2O12",
    "Li6BaLa2Ta2O12",
    "Li6SrLa2Ta2O12",
    "Li7La3Zr2O12",#LLZOGarnet
    "Li6.75La3(Zr1.75Nb0.25)O12",
    "Li7.06La3Y0.06Zr1.94O12",
    "Li6.4Ga0.2La3Zr2O12",
    "Li6.20Ga0.30La2.95Rb0.05Zr2O12",
    "Li0.34La0.51TiO2.94",#Perovskites
    "Li0.35La0.55TiO3",
    "LiSr1.65Zr1.3Ta1.7O9",
    "Li0.375Sr0.4375Ta0.75Zr0.25O3",
    "LiSr1.65Zr1.3Ta1.7O9",
    "Li0.375Sr0.4375Ta0.75Hf0.25O3",
    "Li3OCl", #antiperovskites
    "Li3OCl0.5Br0.5"]

for i, compound in enumerate(test_crystals):
    # compound = "Li1.3Al0.3Ge1.7(PO4)3" # Just for quick tests, comment out
    print(compound)
    comp = ChemHammer(compound)
    tested_results = []

    for x in result:
        try:
            tested_results.append((x[0], comp.levenshtein_dist(x[0]), x[1]))

        except:
            pass

    # sort on similarity score
    sorted_result = sorted(tested_results, key = lambda x : x[1])
    top_ten = sorted_result[:10]
    string_to_write = f"For compound {compound} the top ten closest matches are:\n"
    print(top_ten)
    # Add each of the closest matches
    for j, icsd_code in enumerate(top_ten):
        cursor.execute(f'select formula from data where icsd_code like "%{icsd_code[1]}%"')
        if j == 0:
            db.loc[j, 'Predicted ICSD'] = icsd_code[1]
            db.loc[j, 'Predicted ICSD Formula'] = cursor.fetchall()[0][0]
        cursor.execute(f'select formula from data where icsd_code like "%{icsd_code[1]}%"')
        print_string = f"{icsd_code[1]} : {cursor.fetchall()[0][0]} with distance {icsd_code[0]}"
        string_to_write += print_string + "\n"

    strings.append(string_to_write)


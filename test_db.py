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

# Add a new column for predicted ICSD codes
compounds = db['Compound Formula']
my_compounds, my_splits = split_indices(compounds)

checked = db['Confirmed Y/N']
my_checked = checked[my_splits]

strings = []

print("Enttering main loop")
tested_indices = []

for i, compound in enumerate(my_compounds):
    comp = ChemHammer(compound)
    tested_results = []

    for x in result:
        try:
            tested_results.append((comp.levenshtein_dist(x[0]), x[1]))

        except:
            pass

    # sort on similarity score
    sorted_result = sorted(tested_results, key = lambda x : x[0])
    top_ten = sorted_result[:10]
    string_to_write = f"For compound {compound} the top ten closest matches are:\n"

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

# Once each process has tested theirs send back to root and write to file
sendcounts = np.array(comm.gather(len(strings), root))

if rank == root:
    stringbuf = np.empty(sum(sendcounts), dtype=int)
    indexbuf = np.empty(sum(sendcounts), dtype=int)
else:
    stringbuf = None
    indexbuf = None

comm.Gatherv(sendbuf=strings, recvbuf=(stringbuf, sendcounts), root=root)
comm.Gatherv(sendbuf=my_splits, recvbuf=(indexbuf, sendcounts), root=root)

print("Got to this bit!")
print(len(my_splits))

# Append to dataframe and save excel
db.loc[i, 'Potential ICSD'] = string_to_write

if i % 5 == 0:
    print(f"{i/len(db) * 100}% complete")

writer = ExcelWriter('PythonExport.xlsx')
db.to_excel(writer, 'Sheet1')
writer.save()

print(f"Time taken: {time_start - time()}")


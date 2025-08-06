import pandas as pd
import numpy as np
import math

from collections import defaultdict

# functions used in extract_pet
def calc_traj(row):
    l = row[row[1:].first_valid_index()].split(", ")
    r = row[row.last_valid_index()].split(", ")
    entry = (float(r[0])-float(l[0]),float(r[1])-float(l[1]))
    return entry

def cosine_similarity(a, b):
    if a == (0,0) or b == (0,0):
        return 1
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_traj(df):
    output = []
    for r in range(df.shape[0]):
        output.append(calc_traj(df.iloc[r]))
    return output

def clean_df(df, traj_list, mag_cutoff, valid_classes):
    traj_len = len(traj_list)
    for j in range(traj_len):
        i = traj_len - j - 1
        if np.linalg.norm(traj_list[i]) < mag_cutoff or df.iloc[i, 0] not in valid_classes:
            df.drop(df.index[i], inplace=True)
            traj_list.pop(i)
    return [traj_list, df]

def extract_pet(df, sec_cutoff=4, traj_cutoff=0.9, mag_cutoff=50, valid_classes=["bus","truck","car","motorcycle"]):
    fps = round(1/float(df.columns[2]), 3) # extract FPS
    delta_f = int(sec_cutoff * fps) # change in frames based on FPS and sec_cutoff

    traj_list = generate_traj(df) # generate trajectories for each row of the df for cleaning

    traj_list, df = clean_df(df, traj_list, mag_cutoff, valid_classes) # remove "stationary" objects and nonvehicles

    collisions = defaultdict(lambda: 100) # list of tuples of colliding ids
    for r in range(df.shape[0]-1): # iterating through the valid rows (magnitude > 50)
        row = df.iloc[r] # getting the main row
            
        for c in range(1, len(row)): # iterating through each column in that row
            if not isinstance(row.iloc[c], float): # checking the main row is not nan at that position
                # formating data of row-to-be-checked
                s_entry = row.iloc[c].split(", ")
                entry = [float(coord) for coord in s_entry]
                
                # identifying the rows to compare it to
                checklist = []
                for potrow in range(r+1, df.shape[0]):
                    if (not isinstance(df.iloc[potrow, c], float) and # making sure the other rows aren't nan at this position (bad if low fps)
                        cosine_similarity(traj_list[potrow], traj_list[r]) < traj_cutoff): # making sure the trajectories are different enough
                        
                        checklist.append(potrow)

                for i in checklist: # iterating through each row of the potential rows
                    drow = df.iloc[i]
                    for j in range((c-min(c-2, delta_f)), (c+ min(len(row)-c, delta_f))): # iterating through the columns
                        if not isinstance(drow.iloc[j], float): # checking so it's not nan
                            # formatting data of the row-column entry
                            c_s_entry = drow.iloc[j].split(", ")
                            c_entry = [float(coord) for coord in c_s_entry]
                            
                            # ACTUAL CALCULATIONS/JUDGEMENTS
                            if math.dist(entry, c_entry) <= 20: # checking collision
                                if df.columns[c] - df.columns[j] < collisions[int(df.index[i]), int(df.index[r])]: # checking if PET was lower
                                    collisions[int(df.index[i]), int(df.index[r])] = df.columns[c] - df.columns[j] 

    # collisions are in fomrat (id1, id2)
    return collisions

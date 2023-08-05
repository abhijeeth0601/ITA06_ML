import numpy as np
import pandas as pd

# import csv and convert that csv into dataframe
data = pd.DataFrame(data=pd.read_csv('D:/folders/ML/CSV/enjoysport.csv'))

concepts = np.array(data.iloc[:, 0:-1])
print("Concept\n", concepts)

target = np.array(data.iloc[:, -1])
print("Target\n", target)


def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("\n\ninitialization of specific_h and general_h : \n")
    print("specific_h\n", specific_h, "\n")

    general_h = [["?" for i in range(len(specific_h))]
                 for i in range(len(specific_h))]
    print("General_h\n", general_h, "\n")

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        #             print(specific_h)
        # print(specific_h)
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("\n\nsteps of Candidate Elimination Algorithm", i+1)
        print(specific_h)
        print(general_h)

    indices = [i for i, val in enumerate(general_h) if val ==
               ['?', '?', '?', '?', '?', '?']]

    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h


s_final, g_final = learn(concepts, target)
print("\n\nFinal Specific_h:", s_final, sep="\n")
print("\n\nFinal General_h:", g_final, sep="\n")

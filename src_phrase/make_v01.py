import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pickle.load(open("song_df.pkl","rb"))

x1=[]
x2=[]
for r,l in zip(df["NoteLengths"],df["NoteLabels"]):
    for e1,e2 in zip(r,l):
        x1.append(e1)
        x2.append(e2)
x=np.array(x1)
fig = plt.figure()
#plt.hist(x[x<100000], bins=100, color='red', alpha=0.5)
plt.hist(x, bins=100, color='red', alpha=0.5)
plt.show()

for e1,e2 in zip(x1,x2):
    if e1>40000:
        print(e1,e2)

counter={}
target_sym_list=[]
for e1,e2 in zip(x1,x2):
    if e2 not in counter:
        counter[e2]=0
    else:
        counter[e2]+=1
for k,v in counter.items():
    if v>=100:
        print(k,v)
        target_sym_list.append(k)





import glob
import os
import subprocess
from multiprocessing import Pool

for k in range(0,22,2):
    dest_dir="data2_n{:02d}/".format(k)
    print(dest_dir)
    os.makedirs(dest_dir,exist_ok=True)

cmd_list=[]
for path in glob.glob("./data2/*.wav"):
    name=os.path.basename(path)
    for k in range(0,22,2):
        dest_dir="data2_n{:02d}".format(k)
        noise_ratio=0.1*k
        #subprocess.run(["micarrayx-add-noise",path,dest_dir+"/"+name, "-N", str(noise_ratio)])
        cmd_list.append(["mic",path,dest_dir+"/"+name, "-N","{:.2f}".format(noise_ratio)])
def process(cmd):
    print(cmd)
    subprocess.run(cmd)
print(len(cmd_list))
p = Pool(64)
p.map(process, cmd_list)
p.close()




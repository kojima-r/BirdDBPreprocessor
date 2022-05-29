
import glob
import os
import subprocess
import copy
import shutil

for path in glob.glob("./data/**/**/wavs/*.wav"):
    print(path)
    #./data/WETA/WETA/wavs/2010-05-28_06-42-00-000000.wav
    name=os.path.basename(path)
    src_dir=os.path.dirname(path)
    dest_dir="data_clean/"+src_dir
    dest_path=dest_dir+"/"+name
    os.makedirs(dest_dir,exist_ok=True)
    subprocess.run(["sox",path,"-b","16","-r","44100","-c","1",dest_path])
    ## copy text grid
    arr_dst=path.split("/")
    arr_dst[-2]="TextGrids"
    arr_dst=arr_dst[1:-1]
    arr_src=copy.copy(arr_dst)
    src ="/".join(arr_src)
    dest="data_clean/"+"/".join(arr_dst)
    if os.path.isdir(src) and not os.path.exists(dest):
        print(src,"=>",dest)
        shutil.copytree(src,dest)




    


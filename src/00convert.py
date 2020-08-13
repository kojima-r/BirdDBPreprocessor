
import glob
import os
import subprocess

for path in glob.glob("./data/**/**/wavs/*.wav"):
    print(path)
    #./data/WETA/WETA/wavs/2010-05-28_06-42-00-000000.wav
    name=os.path.basename(path)
    src_dir=os.path.dirname(path)
    dest_dir="data_clean/"+src_dir
    dest_path=dest_dir+"/"+name
    os.makedirs(dest_dir,exist_ok=True)
    subprocess.run(["sox",path,"-b","16","-r","44100","-c","1",dest_path])

    


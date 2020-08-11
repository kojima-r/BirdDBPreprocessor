import glob
from praatio import tgio
import numpy as np
import wave
from scipy.io import wavfile
import pickle
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
from datetime import datetime, timedelta
import os
import click

@click.command()
@click.option('--input_df', '-p', default='song_df.pkl')
@click.option('--output_path', '-o', default='./data2')
def sep_wav(input_df,output_path):
    parallel = True
    verbosity=1
    n_jobs=32
    save=True
    print("[LOAD]",input_df)
    with open(input_df, 'rb') as fp:
        song_df=pickle.load(fp)
    os.makedirs(output_path,exist_ok=True)

    def getSyllsFromWav(row, _mel_basis, WavTime, hparams):
      print("[GET]",row, _mel_basis, WavTime, hparams)
      try:
            rate, data = wavfile.read(row.WavLoc)
      except Exception as e:
          print('WAV file did not load: ' + row.WavLoc)
          print(e)
          return None
      name,_=os.path.splitext(os.path.basename(row.WavLoc))

      for (syll_start, syll_len, syll_sym) in zip(row.NotePositions,
                                                    row.NoteLengths,
                                                    row.NoteLabels):
        syll_stop=syll_start+syll_len
        d=data[syll_start:syll_stop]
        filename=output_path+"/"+name+"."+str(syll_start)+"-"+str(syll_stop)+"."+syll_sym+".wav"
        wavfile.write(filename,rate,d)
      return None

    _mel_basis="_mel_basis"
    hparams="hparams"
    syll_size = 128
    key_list = (
            'all_bird_wav_file', # Wav file (bout_raw) that the syllable came from
            'all_bird_syll', # spectrogram of syllable
            'all_bird_syll_start', # time that this syllable occured
            'all_bird_t_rel_to_file', # time relative to bout file that this 
            'all_bird_syll_lengths', # length of the syllable
            'all_bird_symbol', # the symbolic representation of the syllable
           )
    for indv in tqdm(np.unique(song_df.bird)[::-1]):
        num_notes = np.sum(song_df[song_df.bird ==indv].NumNote)
        if num_notes < 1000: continue
        print(indv, num_notes)
        if parallel:
            with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
                bird_data_packed  = parallel(
                        delayed(getSyllsFromWav)(row, _mel_basis, row.WavTime, hparams) 
                    for idx, row in tqdm(song_df[song_df.bird ==indv].iterrows(),
                                         total=np.sum(song_df.bird ==indv), leave=False))
        else:
            bird_data_packed = [getSyllsFromWav(row, _mel_basis, row.WavTime, hparams) 
                for idx, row in tqdm(song_df[song_df.bird ==indv].iterrows(),
                                      total=np.sum(song_df.bird ==indv), leave=False)]
        

def main():
    sep_wav()

if __name__ == '__main__':
    main()

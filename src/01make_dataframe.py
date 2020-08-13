import glob
from praatio import tgio
import numpy as np
import wave
from scipy.io import wavfile
import pickle
import pandas as pd
#from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
from datetime import datetime, timedelta
import os
import click

@click.command()
@click.option('--input_path', '-p', default='./data_clean/data')
@click.option('--output', '-o', default='song_df.pkl')
def make_dataframe(input_path):
    bird_species = glob.glob(input_path+'/*')

    dataset_sizes = {}
    song_df = pd.DataFrame(columns=['bird', 'species', 'WavTime', 'WavLoc', 'WaveFileName','Position','Length', 'NumNote', 'NotePositions', 'NoteLengths', 'NoteLabels'])
    for species_folder in bird_species:
        species = species_folder.split('/')[-1]
        print(species)
        individuals = glob.glob(species_folder+'/*')
        dataset_sizes[species] = []
        for individual_folder in tqdm(individuals, leave=False):
            individual = individual_folder.split('/')[-1]
            textgrids = glob.glob(individual_folder+'/TextGrids/*.TextGrid')
            for textgrid_loc in tqdm(textgrids, leave=False):
                wav_time = datetime.strptime(textgrid_loc.split('/')[-1][:-9], "%Y-%m-%d_%H-%M-%S-%f")
                # load the textgrid
                print(textgrid_loc)
                try: 
                    tg = tgio.openTextgrid(textgrid_loc)
                except:
                    print('TextGrid did not load')
                    continue
                # extract song from tiers
                all_tiers = [tg.tierDict[tier].entryList for tier in tg.tierDict]
                main_tier = all_tiers[0]
                # create entry for symbolid df
                if len(np.array(main_tier).T) == 0:
                    continue
                start_list, stop_list, label_list = np.array(main_tier).T
                # load the wav
                wav_loc = '/'.join(textgrid_loc.split('/')[:-2] + ['wavs'] + [textgrid_loc.split('/')[-1][:-9]+'.wav'])
                if not os.path.exists(wav_loc): continue
                try:
                    with wave.open(wav_loc, "rb") as wave_file:
                        rate = wave_file.getframerate()
                except:
                    print("[ERROR]",wav_loc)
                    continue
                # create row
                song_df.loc[len(song_df)] = [individual, species, wav_time, wav_loc, wav_loc.split('/')[-1],None, None, len(main_tier),
                                 list((np.array([i.start for i in main_tier])*rate).astype('int')),
                                 list((np.array([i.end - i.start for i in main_tier])*rate).astype('int')),
                                 [i.label for i in main_tier],
                                ]

    filename=output
    with open(filename, 'wb') as fp:
        pickle.dump(song_df,fp)

def main():
    make_dataframe()

if __name__ == '__main__':
    main()


#########################################################
## 01-03 download wav and annotation files from BirdDB ##
#########################################################

python src/00download_data.py
python src/00convert.py

############################
## 01-03 data preparation ##
############################

python src/01make_dataframe.py \
	--input_path  ./data_clean/ \
	--output      ./song_df.pkl 

python src/02sep_wav.py \
	--input_df    ./song_df.pkl \
	--output_path ./data2/

python src/03annotation.py \
	 --input_df   ./song_df.pkl  --input_path ./data2 \
	 --output_df  ./song_df2.pkl --output_csv ./song_df2.csv --output_label label.csv

############################
## 04 feature extractopm  ##
############################

python src/04make_dataset.py \
	--input_df    ./song_df2.pkl \
	--output_path ./data3 \
	--output_ex_path ./data_seq3 \
	--feature mel


python src/04make_dataset.py \
	--input_df    ./song_df2.pkl \
	--output_ex_path ./data_seq3 \
	--output_path ./data3 \
	--feature spec 

python src/04make_dataset.py \
	--input_df    ./song_df2.pkl \
	--output_ex_path ./data_seq3 \
	--output_path ./data3 \
	--feature mel2

python src/04make_dataset.py \
	--input_df    ./song_df2.pkl \
	--output_ex_path ./data_seq3 \
	--output_path ./data3 \
	--feature mfcc


#########################
## 05 embedding        ##
#########################

## frame to emmbedding space
python src/05embedding.py \
	--input_path  ./data3 \
	--output_path out \
	--resample 10000

## sequence to emmbedding space
python src/05embedding.py \
	--input_path ./data_seq3 \
	--output_path out_seq

## sliding window to emmbedding space
python src/05embedding.py --input_path ./data_seq3 --output_path out_seq_win --method umap --feature mel --limit_length 50
python src/05embedding.py --input_path ./data_seq3 --output_path out_seq_win --method trimap --feature mel --limit_length 50
python src/05embedding.py --input_path ./data_seq3 --output_path out_seq_win --method tsne --feature mel --limit_length 50

python src/05embedding.py --input_path ./data_seq3 --output_path out_seq_win --method umap --feature mfcc --limit_length 50
python src/05embedding.py --input_path ./data_seq3 --output_path out_seq_win --method trimap --feature mfcc --limit_length 50
python src/05embedding.py --input_path ./data_seq3 --output_path out_seq_win --method tsne --feature mfcc --limit_length 50


python src/05embedding.py --input_path ./data_seq3 --output_path out_seq_win --method umap --feature mel2 --limit_length 50
python src/05embedding.py --input_path ./data_seq3 --output_path out_seq_win --method trimap --feature mel2 --limit_length 50
python src/05embedding.py --input_path ./data_seq3 --output_path out_seq_win --method tsne --feature mel2 --limit_length 50


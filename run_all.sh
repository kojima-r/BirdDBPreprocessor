python src/01make_dataframe.py \
	--input_path  ./data/ \
	--output      ./song_df.pkl 

python src/02sep_wav.py \
	--input_df    ./song_df.pkl \
	--output_path ./data2/

python src/03annotation.py \
	 --input_df   ./song_df.pkl  --input_path ./data2 \
	 --output_df  ./song_df2.pkl --output_csv ./song_df2.csv --output_label label.csv

python src/04make_dataset.py \
	--input_df    ./song_df2.pkl \
	--output_path ./data3
	--output_ex_path ./data_seq3

python src/05plot.py \
	--input_path  ./data3 \
	--output_path out \
	--resample 10000

python src/05plot.py \
	--input_path ./data_seq3 \
	--output_path out_seq


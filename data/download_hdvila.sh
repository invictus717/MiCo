video2dataset \
--url_list='hd_vila.parquet' \
--input_format='parquet' \
--output_format='files' \
--output_folder="./hdvila" \
--url_col="url" \
--enable_wandb=False \
--encode_formats="{'video': 'mp4', 'audio': 'mp3'}" \
--config="config.yaml" \
--max_shard_retry=3
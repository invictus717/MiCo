torchrun --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node 8 \
    --master_port 9814 \
    run.py \
    --config ./caption_config/caption-generation-audio.json \
    --pretrain_dir './audio_captioner' \
    --output_dir './output/audio_caption' \
    --test_batch_size 128 \
    --generate_nums 3 \
    --captioner_mode true \
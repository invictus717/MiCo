torchrun --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node 8 \
    --master_port 9814 \
    run.py \
    --config ./caption_config/caption-generation-vision.json \
    --pretrain_dir './vision_captioner' \
    --output_dir './output/vision_caption' \
    --test_batch_size 64 \
    --test_vision_sample_num 8 \
    --generate_nums 3 \
    --captioner_mode true \

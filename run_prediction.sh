python prediction.py \
    --batch_input_csv ./datasets/example_feature/feature_path.csv \
    --ckpt_path ./ckpts/ \
    --device cuda:1 \
    --output ./output/prediction_results/result.json \
    --ASM_output_path ./output/ASM \
    --verbose

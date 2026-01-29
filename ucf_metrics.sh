max=500
for i in $(seq 1 $max); do
    echo "\n===============================================\nRun $i/$max"
    nohup  /scratch/s224075134/miniconda/envs/latte/bin/python tools/my_cal_metrics_for_dataset.py --real_data_path /scratch/s224075134/temporal_diffusion/datasets/video/ucf101/images --fake_data_path generation/ucf101_img256/001-MatLatteIMG-XL-256-512-2-F16S3-ucf101_img256-Compile-Amp-loadpixel/ddpm_250/0780000 --resolution 256 --verbose --real-sample-factor 3 --mirror --seed $i
done
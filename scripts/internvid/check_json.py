import json
import pandas as pd

json_path = '/scratch/s224075134/temporal_diffusion/datasets/video/internvid/InternVid-18M-aes.parquet'

df = pd.read_parquet(json_path)
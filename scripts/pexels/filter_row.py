import pandas as pd

JSON_PATH = "/scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/pexels_400k.json"
FILTERED_VIDEO_IDX = [61539]

df = pd.read_json(JSON_PATH)

def filter_row(row):
    if row["video_id"] in FILTERED_VIDEO_IDX:
        return None
    return row

filtered_df = df.apply(filter_row, axis=1).dropna().reset_index(drop=True)
# cast video_id, num_frames, resolution, duration to int
filtered_df["video_id"] = filtered_df["video_id"].astype(int)
filtered_df["num_frames"] = filtered_df["num_frames"].astype(int)
filtered_df["duration"] = filtered_df["duration"].astype(int)
filtered_df["resolution"] = filtered_df["resolution"].apply(lambda x: {"width": int(x["width"]), "height": int(x["height"])})
# append [ at the beginning and ] at the end to make it a valid JSON array
with open(JSON_PATH.replace(".json", "_v1.json"), "w") as f:
    f.write("[\n")
    for i, row in filtered_df.iterrows():
        f.write(row.to_json() + (",\n" if i < len(filtered_df) - 1 else "\n"))
    f.write("]\n")
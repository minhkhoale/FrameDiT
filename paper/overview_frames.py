

frames = [
    '/scratch/s224075134/temporal_diffusion/datasets/video/sky_timelapse/sky_train/2KikwLp-PUI/2KikwLp-PUI_1/2KikwLp-PUI_frames_00000002.jpg',
    '/scratch/s224075134/temporal_diffusion/datasets/video/sky_timelapse/sky_train/2KikwLp-PUI/2KikwLp-PUI_1/2KikwLp-PUI_frames_00000009.jpg',
    '/scratch/s224075134/temporal_diffusion/datasets/video/sky_timelapse/sky_train/2KikwLp-PUI/2KikwLp-PUI_1/2KikwLp-PUI_frames_00000016.jpg',
    '/scratch/s224075134/temporal_diffusion/datasets/video/sky_timelapse/sky_train/2KikwLp-PUI/2KikwLp-PUI_1/2KikwLp-PUI_frames_00000023.jpg',
    '/scratch/s224075134/temporal_diffusion/datasets/video/sky_timelapse/sky_train/2KikwLp-PUI/2KikwLp-PUI_1/2KikwLp-PUI_frames_00000030.jpg',
]

# center crop to square and save to paper/overview_frames
import os
from PIL import Image
output_dir = 'paper/overview_frames'
os.makedirs(output_dir, exist_ok=True)
for frame_path in frames:
    img = Image.open(frame_path)
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    img_cropped = img.crop((left, top, right, bottom))
    base_name = os.path.basename(frame_path)
    output_path = os.path.join(output_dir, base_name)
    img_cropped.save(output_path)
    print(f'Saved cropped image to {output_path}')
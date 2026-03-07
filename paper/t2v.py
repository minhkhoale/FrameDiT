import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from decord import VideoReader


def sample_indices(num_frames, k):
    return np.linspace(0, num_frames - 1, k, dtype=int)


def read_video_frames(video_path, num_samples=6):
    vr = VideoReader(video_path)
    idx = sample_indices(len(vr), num_samples)
    frames = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(f) for f in frames]


def make_strip(frames, height=140, pad=4):
    frames = [f.resize((int(f.width * height / f.height), height)) for f in frames]
    total_w = sum(f.width for f in frames) + pad * (len(frames) - 1)
    canvas = Image.new("RGB", (total_w, height), (255, 255, 255))

    x = 0
    for f in frames:
        canvas.paste(f, (x, 0))
        x += f.width + pad

    return canvas


def plot_t2v_comparison(
    model1_videos,
    model2_videos,
    prompts,
    model1_name="Model 1",
    model2_name="Model 2",
    num_samples=6,
    frame_height=140,
    save_path="comparison.png",
):

    n = len(prompts)

    # height ratios: title small, video big, prompt small
    ratios = [0.0]
    for _ in range(n):
        ratios.extend([2.0, 0.12])

    fig = plt.figure(figsize=(14, 2.3*n))
    gs = fig.add_gridspec(
        n * 2 + 1,
        2,
        height_ratios=ratios
    )

    # titles
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.set_title(model1_name, fontsize=15)
    ax2.set_title(model2_name, fontsize=15)

    ax1.axis("off")
    ax2.axis("off")

    for i in range(n):
        frames1 = read_video_frames(model1_videos[i], num_samples)
        frames2 = read_video_frames(model2_videos[i], num_samples)

        strip1 = make_strip(frames1, frame_height)
        strip2 = make_strip(frames2, frame_height)

        row = 2 * i + 1

        ax_m1 = fig.add_subplot(gs[row, 0])
        ax_m2 = fig.add_subplot(gs[row, 1])

        ax_m1.imshow(strip1)
        ax_m2.imshow(strip2)

        ax_m1.axis("off")
        ax_m2.axis("off")

        # prompt row
        ax_prompt = fig.add_subplot(gs[row + 1, :])
        ax_prompt.text(
            0.5,
            0.9,
            prompts[i],
            ha="center",
            va="center",
            fontsize=14,
            style="italic"
        )
        ax_prompt.axis("off")

    plt.subplots_adjust(hspace=0.15, wspace=0.05)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path[:-4] + '.pdf', dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)


if __name__ == "__main__":
    # Example usage
    model1_videos = [
        "/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/frame/a boat accelerating to gain speed-0_re.mp4",
        #"/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/frame/a bear-4.mp4",
        "/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/frame/a cow bending down to drink water from a river-0.mp4",
        "/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/frame/a train accelerating to gain speed-21.mp4",
        #"/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/frame/A cat wearing sunglasses at a pool-2.mp4"
    ]
    model2_videos = [
        "/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/latte/a boat accelerating to gain speed-0.mp4",
        #"/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/latte/a bear-4.mp4",
        "/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/latte/a cow bending down to drink water from a river-0.mp4",
        "/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/latte/a train accelerating to gain speed-0.mp4",
        #"/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/paper/latte/A cat wearing sunglasses at a pool-2.mp4"
    ]
    prompts = [
        "A boat accelerating to gain speed.",
        #"A bear.",
        "A cow bending down to drink water from a river.",
        "A train accelerating to gain speed.",
        #"A cat wearing sunglasses at a pool."
    ]

    plot_t2v_comparison(
        model1_videos=model1_videos,
        model2_videos=model2_videos,
        prompts=prompts,
        model1_name="FrameDiT-H",
        model2_name="Latte",
        num_samples=4,
        frame_height=190,
        save_path="t2v_comparison.png",
    )
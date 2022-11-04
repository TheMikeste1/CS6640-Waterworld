import imageio


def video_to_gif(video_path, gif_path=None):
    if gif_path is None:
        gif_path = video_path.replace(".mp4", ".gif")
    with imageio.get_reader(video_path, "ffmpeg") as video:
        writer = imageio.get_writer(gif_path, fps=video.get_meta_data()["fps"])
        for frame in video:
            writer.append_data(frame)
        writer.close()


if __name__ == "__main__":
    video_to_gif("test.mp4")

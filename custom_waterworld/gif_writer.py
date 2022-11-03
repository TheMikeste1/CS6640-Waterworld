import imageio


class GIFWriter:
    def __init__(self, fps: int, filename: str):
        if not filename.endswith(".gif"):
            filename += ".gif"
        self.im = imageio.get_writer(filename, mode='I', fps=fps)

    def write(self, frame):
        self.im.append_data(frame)

    def close(self):
        self.im.close()

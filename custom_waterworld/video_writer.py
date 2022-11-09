import os

import cv2


class VideoWriter:
    def __init__(self, fps: int, width: int, height: int, filename: str):
        if not filename.endswith(".mp4"):
            filename += ".mp4"
        name_parts = [
            p
            for parts in map(lambda s: s.split("\\"), filename.split("/"))
            for p in parts
        ]
        filename = os.path.join(
            *name_parts,
        )
        directory = os.path.join(*os.path.split(filename)[:-1])
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        self.vw = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True
        )

    def write(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.vw.write(frame)

    def close(self):
        self.vw.release()

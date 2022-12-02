import os

import imageio
from tqdm.auto import tqdm

from custom_waterworld import GIFWriter

PATH = "./frames/thrust/"

writer = GIFWriter(15, "thrust.gif")
for root, _, files in os.walk(PATH):
    files = sorted(files, key=lambda x: int(x.split(".")[0]))
    for file in tqdm(files):
        path = os.path.join(root, file)
        img = imageio.imread_v2(path)
        writer.write(img)
writer.close()

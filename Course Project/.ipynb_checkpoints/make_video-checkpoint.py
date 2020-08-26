import os
import cv2
from os.path import isfile, join
import re


def generate_video(pathIn = "D:\\MatStatLabs\\Course Project\\pictures_full\\", pathOut="projections_in_area.avi", fps=20):
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort(key = lambda x: int(re.search(r'\d+', x).group(0)))
    frame_array = []

    for i in range(0, len(files), 1):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    height, width, layers = cv2.imread(pathIn + files[0]).shape
    size = (width, height)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


if __name__ == '__main__':
    generate_video(pathOut="centers_mass.avi", fps=2)
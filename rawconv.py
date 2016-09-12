#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import sys
import os.path
import struct

def save_rawimage(img, outpath):
    img = img.convert("F")
    img_seq = [i / 255.0 for i in img.getdata()]
    header = struct.pack("ccxxII", b"P", b"0", img.width, img.height)
    payload = struct.pack("{}f".format(len(img_seq)), *img_seq)
    with open(outpath, "wb") as f:
        f.write(header)
        f.write(payload)


def main():
    if (len(sys.argv) < 2):
        print("Usage: {} file...".format(sys.argv[0]))
        return
    paths = sys.argv[1:]
    for p in paths:
        img = Image.open(p)
        basename = ".".join(p.split(".")[:-1])
        if basename == "":
            basename = p
        save_rawimage(img, "{}_{}x{}_f.dat".format(basename, img.width, img.height))


if __name__ == '__main__':
    main()

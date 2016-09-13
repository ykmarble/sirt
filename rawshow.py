#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import sys
import struct

def load_rawimage(path):
    with open(path, "rb") as f:
        header = struct.unpack("ccxxII", f.read(12))
        if not (header[0] == b"P" and header[1] == b"0"):
            print("Invalied file.")
            sys.exit(1)
        width = header[2]
        height = header[3]
        print(width, height)
        img_seq = struct.unpack("{}f".format(width*height), f.read())
    img = Image.new("F", (width, height))
    img.putdata(img_seq)
    return img

def main():
    if (len(sys.argv) != 2):
        print("Usage: {} file".format(sys.argv[0]))
        return
    path = sys.argv[1]
    img = load_rawimage(path)
    img.show()

if __name__ == '__main__':
    main()

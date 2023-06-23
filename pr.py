#!/usr/bin/env python

import argparse
import glob
import cv2
import PIL.Image as Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="source dir")
    parser.add_argument("--dst", help="destination dir")
    args = parser.parse_args()
    return args.src, args.dst


def get_chapter_name(filename: str):
    return filename[59:-31]


if __name__ == "__main__":
    src_dir, dst_dir = parse_args()
    name_pattern = f"{src_dir}/Screenshot*Permanent Residence Portal.png"
    counter = 0
    pil_images = []
    out_name = None
    for filename in glob.glob(name_pattern):
        print(f"reading {filename}")
        out_name = get_chapter_name(filename)
        image = cv2.imread(filename)
        h, w = image.shape[:2]
        scale = 1500 / w
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        h, w = image.shape[:2]
        h_step = round(w * 1.41428)
        n_pages = max(1, round(h / h_step))
        for i in range(n_pages):
            start = (i * h) // n_pages
            end = ((i + 1) * h) // n_pages
            page = image[start:end, :, :]
            page_rgb = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(page_rgb))
            # out_file = f"{dst_dir}/{out_name}_p{counter}.png"
            # print(f"writing {out_file}")
            # cv2.imwrite(out_file, page)
            counter += 1
    out_file_pdf = f"{dst_dir}/{out_name}.pdf"
    print(f"writing {out_file_pdf}")
    pil_images[0].save(out_file_pdf, "PDF", resolution=300, save_all=True, append_images=pil_images[1:])

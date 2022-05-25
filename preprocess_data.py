import argparse
import PIL
import os
import os.path
from PIL import Image
from tqdm import tqdm

NEW_SHAPE = (640, 360)


def main(args):
    arguments = args.parse_args()
    fold_in = arguments.input_dir
    fold_out = arguments.output_dir
    os.makedirs(fold_out, exist_ok=True)
    for file in tqdm(os.listdir(fold_in), desc="Resizing data:"):
        f_img_in = fold_in + "/" + file
        f_img_out = fold_out + "/" + file
        img = Image.open(f_img_in)
        img = img.resize(NEW_SHAPE)
        img.save(f_img_out)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Preprocessor")
    args.add_argument(
        "-i",
        "--input_dir",
        default=None,
        type=str,
        help="Dir of images",
    )

    args.add_argument(
        "-o",
        "--output_dir",
        default=None,
        type=str,
        help="path to output dir",
    )
    main(args)

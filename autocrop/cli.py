import argparse
import os
import shutil
import sys
from typing import Optional

from PIL import Image

from .__version__ import __version__
from .autocrop import Cropper, ImageReadError
from .constants import (
    QUESTION_OVERWRITE,
    INPUT_FILETYPES,
)


def output(input_filename, output_filename, image):
    """
    Move the input file to the output location and write over it with the
    cropped image data.
    """
    if input_filename != output_filename:
        # Move the file to the output directory
        shutil.copy(input_filename, output_filename)
    # Encode the image as an in-memory PNG
    img_new = Image.fromarray(image)
    # Write the new image (converting the format to match the output
    # filename if necessary)
    img_new.save(output_filename)


def reject(input_filename, reject_filename):
    """Move the input file to the reject location."""
    if input_filename != reject_filename:
        # Move the file to the reject directory
        shutil.copy(input_filename, reject_filename)


def main(
    input_d: str,
    output_d: str,
    reject_d: str,
    extension: Optional[str] = None,
    fheight: int = 500,
    fwidth: int = 500,
    facePercent: int = 50,
    resize: bool = False,
    cropmix: float = 0.5,
) -> None:
    """
    Crops folder of images to the desired height and width if a
    face is found.

    If `input_d == output_d` or `output_d is None`, overwrites all files
    where the biggest face was found.

    Parameters:
    -----------

    - `input_d`: `str`
        * Directory to crop images from.
    - `output_d`: `str`
        * Directory where cropped images are placed.
    - `reject_d`: `str`
        * Directory where images that cannot be cropped are placed.
    - `fheight`: `int`, default=`500`
        * Height (px) to which to crop the image.
    - `fwidth`: `int`, default=`500`
        * Width (px) to which to crop the image.
    - `facePercent`: `int`, default=`50`
        * Percentage of face from height.
    - `extension` : `str`
        * Image extension to save at output.
    - `resize`: `bool`, default=`True`
        * If `False`, don't resize the image, but use the original size.

    Side Effects:
    -------------

    - Creates image files in output directory.
    """
    reject_count = 0
    output_count = 0
    input_files = [
        os.path.join(input_d, f)
        for f in os.listdir(input_d)
        if any(f.endswith(t) for t in INPUT_FILETYPES)
    ]
    if output_d is None:
        output_d = input_d
    if reject_d is None and output_d is None:
        reject_d = input_d
    if reject_d is None:
        reject_d = output_d

    # Guard against calling the function directly
    input_count = len(input_files)
    assert input_count > 0

    # Main loop
    cropper = Cropper(
        width=fwidth, height=fheight, face_percent=facePercent, resize=resize, cropmix=cropmix,
    )
    for input_filename in input_files:
        basename = os.path.basename(input_filename)
        if extension:
            basename_noext = os.path.splitext(basename)[0]
            output_filename = os.path.join(output_d, basename_noext + "." + extension)
        else:
            output_filename = os.path.join(output_d, basename)
        reject_filename = os.path.join(reject_d, basename)
        image = None

        # Attempt the crop
        try:
            image = cropper.crop(input_filename)
        except ImageReadError:
            print("Read error:       {}".format(input_filename))
            continue

        # Did the crop produce an invalid image?
        if isinstance(image, type(None)):
            reject(input_filename, reject_filename)
            print("No face detected: {}".format(reject_filename))
            reject_count += 1
        else:
            output(input_filename, output_filename, image)
            print("Face detected:    {}".format(output_filename))
            output_count += 1

    # Stop and print status

    print(
        f"{input_count} : Input files, {output_count} : Faces Cropped, {reject_count}"
    )


def input_path(p):
    """Returns path, only if input is a valid directory."""
    no_folder = "Input folder does not exist"
    no_images = "Input folder does not contain any image files"
    p = os.path.abspath(p)
    if not os.path.isdir(p):
        raise argparse.ArgumentTypeError(no_folder)
    filetypes = {os.path.splitext(f)[-1] for f in os.listdir(p)}
    if not any(t in INPUT_FILETYPES for t in filetypes):
        raise argparse.ArgumentTypeError(no_images)
    else:
        return p


def output_path(p):
    """
    Returns path, if input is a valid directory name.
    If directory doesn't exist, creates it.
    """
    p = os.path.abspath(p)
    if not os.path.isdir(p):
        os.makedirs(p)
    return p


def size(i):
    """Returns valid only if input is a positive integer under 1e5"""
    error = "Invalid pixel size"
    try:
        i = int(i)
    except ValueError:
        raise argparse.ArgumentTypeError(error)
    if i > 0 and i < 1e5:
        return i
    else:
        raise argparse.ArgumentTypeError(error)

def Notsize(i):
    """Returns valid only if input is a positive integer under 1e5"""
    error = "Invalid pixel size"
    try:
        i = int(i)
    except ValueError:
        raise argparse.ArgumentTypeError(error)
    if i > -0.99 and i < 101:
        return i
    else:
        raise argparse.ArgumentTypeError(error)



def compat_input(s=""):  # pragma: no cover
    """Compatibility function to permit testing for Python 2 and 3."""
    try:
        return raw_input(s)
    except NameError:
        # Py2 raw_input() renamed to input() in Py3
        return input(s)  # lgtm[py/use-of-input]


def confirmation(question):
    """Ask a yes/no question via standard input and return the answer."""
    yes_list = ["yes", "y"]
    no_list = ["no", "n"]
    default_str = "[Y]/n"
    prompt_str = "{} {} ".format(question, default_str)

    while True:
        choice = compat_input(prompt_str).lower()

        if not choice:
            return default_str
        if choice in yes_list:
            return True
        if choice in no_list:
            return False

        notification_str = "Please respond with 'y' or 'n'"
        print(notification_str)


def chk_extension(extension):
    """Check if the extension passed is valid or not."""
    error = "Invalid image extension"
    extension = str(extension).lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    if extension in INPUT_FILETYPES:
        return extension.lower().replace(".", "")
    else:
        raise argparse.ArgumentTypeError(error)


def parse_args(args):
    """Helper function. Parses the arguments given to the CLI."""
    help_d = {
        "desc": "Automatically crops faces from batches of pictures",
        "input": """Folder where images to crop are located. Default:
                     current working directory""",
        "output": """Folder where cropped images will be moved to.

                      Default: current working directory, meaning images are
                      cropped in place.""",
        "reject": """Folder where images that could not be cropped will be
                       moved to.

                      Default: current working directory, meaning images that
                      are not cropped will be left in place.""",
        "extension": "Enter the image extension which to save at output",
        "width": "Width of cropped files in px. Default=500",
        "height": "Height of cropped files in px. Default=500",
        "y": "Bypass any confirmation prompts",
        "facePercent": "Percentage of face to image height",
        "no_resize": """Do not resize images to the specified width and height,
                      but instead use the original image's pixels.""",
        "cropmix": "cropmix between 0-100, 0 being full face crop, 1 being center crop",
    }

    parser = argparse.ArgumentParser(description=help_d["desc"])
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s version {}".format(__version__),
    )
    parser.add_argument(
        "--no-confirm",
        "--skip-prompt",
        action="store_true", help=help_d["y"]
    )
    parser.add_argument(
        "-n",
        "--no-resize",
        action="store_true",
        help=help_d["no_resize"],
    )
    parser.add_argument(
        "-i", "--input", default=".", type=input_path, help=help_d["input"]
    )
    parser.add_argument(
        "-o",
        "--output",
        "-p",
        "--path",
        type=output_path,
        default=None,
        help=help_d["output"],
    )
    parser.add_argument(
        "-r", "--reject", type=output_path, default=None, help=help_d["reject"]
    )
    parser.add_argument("-w", "--width", type=size, default=500, help=help_d["width"])
    parser.add_argument("-H", "--height", type=size, default=500, help=help_d["height"])
    parser.add_argument(
        "--facePercent", type=size, default=50, help=help_d["facePercent"]
    )
    parser.add_argument(
        "-e", "--extension", type=chk_extension, default=None, help=help_d["extension"]
    )
    parser.add_argument(
        "--cropmix", "-c", type=size, default=0.5, help=help_d["cropmix"]
    )    

    return parser.parse_args()


def command_line_interface():
    """
    AUTOCROP
    --------
    Crops faces from batches of images.
    """
    args = parse_args(sys.argv[1:])
    if not args.no_confirm:
        if args.output is None or args.input == args.output:
            if not confirmation(QUESTION_OVERWRITE):
                sys.exit()
    if args.input == args.output:
        args.output = None
    print("Processing images in folder:", args.input)

    resize = not args.no_resize
    main(
        args.input,
        args.output,
        args.reject,
        args.extension,
        args.height,
        args.width,
        args.facePercent,
        resize,
        args.cropmix,
    )

import itertools

import cv2
import numpy as np
import os
import sys
from PIL import Image
from numpy.linalg import norm
from shapely.geometry import LineString as shLs
from shapely.geometry import Point as shPt
from cropimage import Cropper

cropski = Cropper()

from .constants import (
    MINFACE,
    GAMMA_THRES,
    GAMMA,
    CASCFILE,
)


class ImageReadError(BaseException):
    """Custom exception to catch an OpenCV failure type."""

    pass


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def intersect(v1, v2):
    a1, a2 = v1
    b1, b2 = v2
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db).astype(float)
    num = np.dot(dap, dp)
    return (num / denom) * db + b1


def distance(pt1, pt2):
    """Returns the euclidian distance in 2D between 2 pts."""
    distance = np.linalg.norm(pt2 - pt1)
    return distance


def bgr_to_rbg(img):
    """Given a BGR (cv2) numpy array, returns a RBG (standard) array."""
    # Don't do anything for grayscale images
    #if img.ndim == 2:
    #    return img

    # Flip the channels. Use explicit indexing in case RGBA is used.
    img[0, 1, 2] = img[2, 1, 0]
    return img


def gamma(img, correction):
    """Simple gamma correction to brighten faces"""
    img = cv2.pow(img / 255.0, correction)
    return np.uint8(img * 255)


def check_underexposed(image, gray):
    """
    Returns the (cropped) image with GAMMA applied if underexposition
    is detected.
    """
    uexp = cv2.calcHist([gray], [0], None, [256], [0, 256])
    if sum(uexp[-26:]) < GAMMA_THRES * sum(uexp):
        image = gamma(image, GAMMA)
    return image


def check_positive_scalar(num):
    """Returns True if value if a positive scalar."""
    if num > 0 and not isinstance(num, str) and np.isscalar(num):
        return int(num)
    raise ValueError("A positive scalar is required")


def open_file(input_filename):
    """Given a filename, returns a numpy array"""
    with Image.open(input_filename) as img_orig:
        return np.array(img_orig)

class Cropper:
    """
    Crops the largest detected face from images.

    This class uses the `CascadeClassifier` from OpenCV to
    perform the `crop` by taking in either a filepath or
    Numpy array, and returning a Numpy array. By default,
    also provides a slight gamma fix to lighten the face
    in its new context.

    Parameters:
    -----------

    * `width` : `int`, default=500
        - The width of the resulting array.
    * `height` : `int`, default=`500`
        - The height of the resulting array.
    * `face_percent`: `int`, default=`50`
        - Aka zoom factor. Percent of the overall size of
        the cropped image containing the detected coordinates.
    * `fix_gamma`: `bool`, default=`True`
        - Cropped faces are often underexposed when taken
        out of their context. If under a threshold, sets the
        gamma to 0.9.
    * `resize`: `bool`, default=`True`
        - Resizes the image to the specified width and height,
        otherwise, returns the original image pixels.
    """

    def __init__(
        self,
        width=500,
        height=500,
        face_percent=1,
        padding=None,
        fix_gamma=True,
        resize=False,
        cropmix=0.5,
    ):
        self.height = check_positive_scalar(height)
        self.width = check_positive_scalar(width)
        self.aspect_ratio = width / height
        self.gamma = fix_gamma
        self.resize = resize
        self.cropmix = cropmix

        # Face percent
        if face_percent > 100 or face_percent < 1:
            fp_error = "The face_percent argument must be between 1 and 100"
            raise ValueError(fp_error)
        self.face_percent = check_positive_scalar(face_percent)

        # XML Resource
        directory = os.path.dirname(sys.modules["autocrop"].__file__)
        self.casc_path = os.path.join(directory, CASCFILE)

    def crop(self, path_or_array):
        """
        Given a file path or np.ndarray image with a face,
        returns cropped np.ndarray around the largest detected
        face.

        Parameters
        ----------
        - `path_or_array` : {`str`, `np.ndarray`}
            * The filepath or numpy array of the image.

        Returns
        -------
        - `image` : {`np.ndarray`, `None`}
            * A cropped numpy array if face detected, else None.
        """
        if isinstance(path_or_array, str):
            image = open_file(path_or_array)
            imagepath = path_or_array
        else:
            image = path_or_array



        # Some grayscale color profiles can throw errors, catch them
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = image

        # Scale the image
        try:
            img_height, img_width = image.shape[:2]
        except AttributeError:
            raise ImageReadError
        minface = int(np.sqrt(img_height**2 + img_width**2) / MINFACE)

        # Create the haar cascade
        face_cascade = cv2.CascadeClassifier(self.casc_path)

        # ====== Detect faces in the image ======
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(minface, minface),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,
        )

        # Handle no faces

        def bruh(input_filename):
            image = input_filename

        if len(faces) == 0:
            image = cropski.crop(imagepath, target_size=(512,512))
            print("nofacebruh")
            imgeez = image
            image = cv2.cvtColor(imgeez, cv2.COLOR_RGB2BGR)
            return image

        #if img_height is 512:
        #  h = 512
        #elif img_width is 512:
        #  w = 512


        # Make padding from biggest face found
        x, y, w, h = faces[-1]
        pos = self._crop_positions(
            img_height,
            img_width,
            x,
            y,
            w,
            h,
        )



        print(img_height, "img_height")
        print(img_width, "img_width")
        x, y, w, h = faces[-1]

        if img_height > img_width:
            taller = True
            ux1 = (img_width/2)
            uy1 = (512/2)
            uy2 = (img_height - (512/2))
            ux2 = (ux1)
            #lp1 = (int(ux1), int(uy1))
            #lp2 = (int(ux2), int(uy2))
            #lp1 = np.array([int(ux1),int(uy1)])
            lp1 = (ux1, uy1)
            print(lp1)
            #lp2 = np.array([int(ux2), int(uy2)])
            lp2 = (ux2, uy2)
            print(lp2)
            #fp3 = np.array([x + int(w / 2), y + int(h / 2)])
            fp3 = (int(x + (w / 2)), int(y + (h / 2)))
            print(fp3, "fp3")
            #newcenter=np.cross(p2-p1,p3-p1)/norm(p2-p1)
            #newcenter = norm(np.cross(lp2-lp1, lp1-fp3))/norm(lp2-lp1)
            #newcenter = norm(np.cross(int(lp2-lp1, lp1-fp3)))/norm(int(lp2-lp1))


        elif img_height < img_width:
            taller = False
            ux1 = (512/2)
            uy1 = (img_height/2)
            ux2 = (img_width - (512/2))
            uy2 = (uy1)
            #lp1 = np.array([int(ux1),int(uy1)])
            lp1 = (ux1, uy1)
            print(lp1)
            #lp2 = np.array([int(ux2), int(uy2)])
            lp2 = (ux2, uy2)
            print(lp2)
            #fp3 = np.array([x + int(w / 2), y + int(h / 2)])
            fp3 = ((x + (w / 2)), (y + (h / 2)))
            print(fp3, "fp3")
            #newcenter = norm(np.cross(lp2-lp1, lp1-fp3))/norm(lp2-lp1)
            #newcenter = norm(np.cross(int(lp2-lp1, lp1-fp3)))/norm(int(lp2-lp1))
            #newcenter=np.cross(p2-p1,p3-p1)/norm(p2-p1)

        else:
            print("BRUH MOMENT")
            self.cropmix=100
            taller = False
            ux1 = (512/2)
            uy1 = (img_height/2)
            ux2 = (img_width - (512/2))
            uy2 = (uy1)
            #lp1 = np.array([int(ux1),int(uy1)])
            lp1 = (ux1, uy1)
            print(lp1)
            #lp2 = np.array([int(ux2), int(uy2)])
            lp2 = (ux2, uy2)
            print(lp2)
            #fp3 = np.array([x + int(w / 2), y + int(h / 2)])
            fp3 = ((x + (w / 2)), (y + (h / 2)))
            print(fp3, "fp3")
            #newcenter = norm(np.cross(lp2-lp1, lp1-fp3))/norm(lp2-lp1)
            #newcenter = norm(np.cross(int(lp2-lp1, lp1-fp3)))/norm(int(lp2-lp1))
            #newcenter=np.cross(p2-p1,p3-p1)/norm(p2-p1)

        boundline = shLs([ ((ux1), (uy1)), ((ux2), (uy2))])
        print(boundline)
        facepoint = shPt(fp3)
        print(facepoint)
        dist = facepoint.distance(boundline)
        #1.0
        newcenter = boundline.interpolate(dist)
        print("newcenter", newcenter)

        #newcenter = [int(newcenter.x), int(newcenter.y)]
        #print("newcenterasarray", newcenter)
        #npa, npb = newcenter
        npa = int(newcenter.x)
        npb = int(newcenter.y)

        #if taller is True:
        #    if npb <= uy1:
        #        npb = uy1
        #    else:
        #        pass
        #elif taller is False:
        #    if npa <= ux1:
        #        npa = ux1
        #else:
        #    print("BAD")
        #    pass



        newcenter = [int(npa), int(npb)]

        #'POINT (2 1)'

        #def newcenter(lp1, lp2, fp3):
        #timeski=0
        #ux1, uy1 = lp1
        #timeski+=1 #1
        #print(timeski,ux1, uy1)

        #ux2, uy2 = lp2
        #timeski+=1 #2
        #print(timeski, ux2, uy2)

        #fx3, fy3 = fp3
        #timeski+=1 #3
        #print(timeski, fx3, fy3)

        #dtx, dty = ux2-ux1, uy2-uy1
        #timeski+=1 #4
        #print(timeski, dtx, dty)

        #det = dtx*dtx + dty*dty
        #timeski+=1 #5
        #print(timeski, det)

        #a = (dty*(fy3-uy1)+dtx*(fx3-ux1))/det
        #timeski+=1 #6
        #print(timeski, a)
        #newcenter = np.asarray([int(ux1+a*dtx), int(uy1+a*dty)])

        #ncx = ux1+a*dtx
        #timeski+=1 #7
        #print(timeski, ncx)

        #ncy = uy1+a*dty
        #timeski+=1 #8
        #print(timeski, ncy)

        #newcenter = [int(ncx), int(ncy)]
        #timeski+=1 #9
        #print(timeski, newcenter)
        #newcenter = newcenter.shape
        # ====== Actual cropping ======
        #image = image[pos[0] : pos[1], pos[2] : pos[3]]

        #newcenter = np.array[(newcenter(1), )
        #newcenter = newcenter.shape
        #npa x
        #npb y
        print("npa",npa, "npb", npb)
        rowtopski = npb - 256
        rowbottomski = npb + 256
        columnleftski = npa - 256
        columnrightski = npa + 256


        centercroppington = image.shape
        r = centercroppington[1]/2 - 512/2
        u = centercroppington[0]/2 - 512/2

        #cropmix = 1/100
        #dummy = dummy[int(((u)*cropmix)+((fcy) * (1 - cropmix))) : int((((u) + 512) * cropmix) + ((fcy) + 512) * (1 - cropmix)), int(((r) * cropmix) + ((fcx) * (1 - cropmix))) : int(((r+512) * cropmix) + ((fcx)+512) * (1 - cropmix))]
        #print("cropmix at lowest", dummy)
        #cropmix = 100/100
        #dummy = dummy[int(((u)*cropmix)+((fcy) * (1 - cropmix))) : int((((u) + 512) * cropmix) + ((fcy) + 512) * (1 - cropmix)), int(((r) * cropmix) + ((fcx) * (1 - cropmix))) : int(((r+512) * cropmix) + ((fcx)+512) * (1 - cropmix))]
        #print("cropmix at full", dummy)
        cropmix = 50/100
        print("cropmix", cropmix)

        ####image = image[int(((u)*cropmix)+((fcy) * (1 - cropmix))) : int((((u) + 512) * cropmix) + ((fcy) + 512) * (1 - cropmix)), int(((r) * cropmix) + ((fcx) * (1 - cropmix))) : int(((r+512) * cropmix) + ((fcx)+512) * (1 - cropmix))]
        image = image[int(((u)*cropmix)+((rowtopski) * (1 - cropmix))) : int((((u) + 512) * cropmix) + (rowbottomski) * (1 - cropmix)), int(((r) * cropmix) + ((columnleftski) * (1 - cropmix))) : int(((r+512) * cropmix) + (columnrightski) * (1 - cropmix))]

        #crop_img = img[int(y):int(y+h), int(x):int(x+w)]
        #         image = image[pos[0] : pos[1], pos[2] : pos[3]]
        #          crop_img = img[int(y):int(y+h), int(x):int(x+w)]
        #centercroppington = np.array([x + int(w / 2), y + int(h / 2)])

        #x = centercroppington[1]/2 - w/2
        #y = centercroppington[0]/2 - h/2
        #cropmix = self.cropmix/100
        #image = image[int(((u)*cropmix)+(pos[0] * (1 - cropmix))) : int(((u + 512) * cropmix) + (pos[1] * (1 - cropmix))), int(((r) * cropmix) + (pos[2] * (1 - cropmix))) : int(((r+512) * cropmix) + (pos[3] * (1 - cropmix)))]

        #image = image[int(((y)*cropmix)+(pos[0] * (1 - cropmix))) : int(((y + 512) * cropmix) + (pos[1] * (1 - cropmix))), int(((x) * cropmix) + (pos[2] * (1 - cropmix))) : int(((x+512) * cropmix) + (pos[3] * (1 - cropmix)))]

        #         image = image[pos[0] : pos[1], pos[2] : pos[3]]
        #          crop_img = img[int(y):int(y+h), int(x):int(x+w)]

        # Resize
        if self.resize:
            with Image.fromarray(image) as img:
                image = np.array(img.resize((self.width, self.height)))

        # Underexposition fix
        if self.gamma:
            with Image.fromarray(image) as img:
                image = check_underexposed(image, gray)
        return bgr_to_rbg(image)

    def _determine_safe_zoom(self, imgh, imgw, x, y, w, h):
        """
        Determines the safest zoom level with which to add margins
        around the detected face. Tries to honor `self.face_percent`
        when possible.

        Parameters:
        -----------
        imgh: int
            Height (px) of the image to be cropped
        imgw: int
            Width (px) of the image to be cropped
        x: int
            Leftmost coordinates of the detected face
        y: int
            Bottom-most coordinates of the detected face
        w: int
            Width of the detected face
        h: int
            Height of the detected face

        Diagram:
        --------
        i / j := zoom / 100

                  +
        h1        |         h2
        +---------|---------+
        |      MAR|GIN      |
        |         (x+w, y+h)|
        |   +-----|-----+   |
        |   |   FA|CE   |   |
        |   |     |     |   |
        |   ├──i──┤     |   |
        |   |  cen|ter  |   |
        |   |     |     |   |
        |   +-----|-----+   |
        |   (x, y)|         |
        |         |         |
        +---------|---------+
        ├────j────┤
                  +
        """


        #if w > 15:
        #  w=300
        #else:
        #  pass
        #if h > 15:
        #  h=300
        #else:
        #  pass

        # Find out what zoom factor to use given self.aspect_ratio
        corners = itertools.product((x, x + w), (y, y + h))
        center = np.array([x + int(w / 2), y + int(h / 2)])
        i = np.array(
            [(0, 0), (0, imgh), (imgw, imgh), (imgw, 0), (0, 0)]
        )  # image_corners
        image_sides = [(i[n], i[n + 1]) for n in range(4)]

        #if w > 15:
        #  w=50
        #else:
        #  pass
        #if h > 15:
        #  h=50
        #else:
        #  pass

        corner_ratios = [self.face_percent]  # Hopefully we use this one
        for c in corners:
            corner_vector = np.array([center, c])
            a = distance(*corner_vector)
            intersects = list(intersect(corner_vector, side) for side in image_sides)
            for pt in intersects:
                if (pt >= 0).all() and (pt <= i[2]).all():  # if intersect within image
                    dist_to_pt = distance(center, pt)
                    corner_ratios.append(100 * a / dist_to_pt)
        return max(corner_ratios)

    def _crop_positions(
        self,
        imgh,
        imgw,
        x,
        y,
        w,
        h,
    ):
        """
        Retuns the coordinates of the crop position centered
        around the detected face with extra margins. Tries to
        honor `self.face_percent` if possible, else uses the
        largest margins that comply with required aspect ratio
        given by `self.height` and `self.width`.

        Parameters:
        -----------
        imgh: int
            Height (px) of the image to be cropped
        imgw: int
            Width (px) of the image to be cropped
        x: int
            Leftmost coordinates of the detected face
        y: int
            Bottom-most coordinates of the detected face
        w: int
            Width of the detected face
        h: int
            Height of the detected face
        """
        zoom = self._determine_safe_zoom(imgh, imgw, x, y, w, h)

        #img_height, img_width = image.shape[:2]
        #if imgh == 512:
        #  #h = 512
        #  height_crop = 512
        #  width_crop = self.aspect_ratio * float(height_crop)
        #  print("h",self.height, h)

        #elif imgw == 512:
          #w = 512
        #  width_crop = 512
        #  height_crop = float(width_crop) / self.aspect_ratio

        #  print("w",self.width, w)
        #else:
        #  print("BOO",self.width, w, imgw)
        #  print("BOO",self.height, h, imgh)

        # Adjust output height based on percent
        if self.height >= self.width:
            height_crop = w * 100 / zoom
            width_crop = self.aspect_ratio * float(height_crop)
        else:
            width_crop = w * 100 / zoom
            height_crop = float(width_crop) / self.aspect_ratio

        # Calculate padding by centering face
        xpad = (width_crop - w) / 2
        ypad = (height_crop - h) / 2

        # Calc. positions of crop
        h1 = x - xpad
        h2 = x + w + xpad
        v1 = y - ypad
        v2 = y + h + ypad

        return [int(v1), int(v2), int(h1), int(h2)]

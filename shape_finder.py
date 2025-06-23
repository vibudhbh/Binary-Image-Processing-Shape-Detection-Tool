import numpy as np
import cv2 as cv
import os
from typing import Optional
from enum import Enum


class Size(Enum):
    LARGE = "large"
    SMALL = "small"

    def __str__(self):
        return self.value


class Color(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    CYAN = "cyan"
    BLUE = "blue"
    MAGENTA = "magenta"

    def __str__(self):
        return self.value


class Shape(Enum):
    CIRCLE = "circle"
    WEDGE = "wedge"
    RECTANGLE = "rectangle"
    CROSS = "cross"

    def __str__(self):
        return self.value


COLOR_TO_HUE_LOOKUP = dict(
    zip(Color, [0, 30, 60, 90, 120, 150])
)  # changed every value to match hue range of respective color
MIN_AREA = 100  # changed from 1000
HUE_TOLERANCE = 10


def otsu_threshold(counts: np.ndarray, bins: Optional[np.ndarray] = None) -> float:
    """Given a histogram (a numpy array where counts[i] is # of values where x=bins[i]) return
    the threshold that minimizes intra-class variance using Otsu's method. If 'bins' is provided,
    then the x-coordinate of counts[i] is set by bins[i]. Otherwise, it is assumed that the
    x-coordinates are 0, 1, 2, ..., len(counts)-1. If provided, 'bins' must be sorted in
    ascending order and have the same length as 'counts'.

    Note: For didactic purposes, this function uses numpy only and does not rely on OpenCV.
    """
    if bins is not None:
        if not len(counts) == len(bins):
            raise ValueError("bins must have the same length as counts")
        if not np.all(bins[:-1] <= bins[1:]):
            raise ValueError("bins must be sorted in ascending order")
    else:
        bins = np.arange(len(counts))

    def variance_helper(bins_: np.ndarray, counts_: np.ndarray) -> float:
        n = np.sum(counts_)
        if n == 0:
            return 0
        mu = np.dot(bins_, counts_) / n
        return np.dot(counts_, (bins_ - mu) ** 2) / n

    lowest_variance, best_threshold = float("inf"), 0
    for i in range(len(counts) - 1):  # changed from len(counts) to not get index out of range error
        variance_left = variance_helper(bins[: i + 1], counts[: i + 1])
        variance_right = variance_helper(bins[i + 1 :], counts[i + 1 :])

        total_variance = variance_left + variance_right  # added to calculate total variance
        if (
            total_variance < lowest_variance
        ):  # changed from comparing variance_left > variance_right
            # Set the threshold to the midpoint between bins[i] and bins[i+1]
            th = (bins[i] + bins[i + 1]) / 2  # changed to always return midpoint
            lowest_variance, best_threshold = (
                total_variance,
                th,
            )  # changed lowest_variance to total_variance
    return best_threshold


def roundedness(moments: dict[str, float]) -> float:
    """Given the moments of a shape, return the roundedness of the shape. The roundedness is
    defined as the ratio of two second-moment axes to each other. The circle and cross shapes have
    a roundedness of close to 1, while the rectangle and wedge shapes have a roundedness less than 1

    Note: see the OpenCV docs for the difference between moments["mu20"] and moments["m20"]. The
    latter is in the lecture slides, but the former makes the calculation easier.
    https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
    """
    covariance = np.array([[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]])
    # The eigenvalues of the covariance matrix are the variances of the shape in the x and y
    # directions. The roundedness is the ratio of the smallest standard deviation to the largest.
    stdevs = np.sqrt(np.linalg.eigvalsh(covariance))
    return min(stdevs) / max(stdevs)  # added division by max(stdevs)


def threshold_on_hue(image: np.ndarray, color: Color, hue_tolerance: int = 10) -> np.ndarray:
    """The job of this function is to convert the image to binary form, where the shapes of a
    particular color are 1 and the background is 0. By 'color' we mean that the hue of a pixel is
    equal to the hue named by the given color string, plus or minus hue_tolerance. This is done
    by thresholding the image, and applying some morphological operations to clean up the result.
    """

    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Choose a saturation threshold using Otsu's method so that we select only the most saturated
    # pixels as ones.
    saturation_hist = (
        cv.calcHist([hsv], [1], None, [255], [1, 256]).astype(np.float32).ravel()
    )  # changed channels from [0] to [1], and range from [0, 256] to [1, 256]

    saturation_lo, saturation_hi = otsu_threshold(saturation_hist), 255
    # Debugging
    # import matplotlib.pyplot as plt

    # Plot saturation histogram
    # plt.plot(saturation_hist)
    # plt.title('Saturation Histogram')
    # plt.show()

    # Debugging: print the saturation threshold
    # print(f"Saturation threshold: {saturation_lo}")

    # Any value
    value_lo, value_hi = 0, 255

    # We're going to threshold on both hue and saturation. The saturation range will be any value
    # above saturation_lo. The hue range will be set by the color plus or minus hue_tolerance.
    reference_hue = COLOR_TO_HUE_LOOKUP[color]

    # Debugging
    # print(f"Reference hue for {color}: {reference_hue}")
    # print(f"Hue tolerance for {color}: {hue_tolerance}")

    if hue_tolerance >= 90:
        # A tolerance of plus or minus 90 on a range from 0 to 180 means "keep everything"
        hue_ranges = [(0, 180)]
    elif reference_hue - hue_tolerance < 0:
        # Handle the case where we wrapped around left. We need two ranges: one from 0 to the
        # reference hue + tolerance, and one covering the amount overlapped on the 180 side.
        hue_ranges = [
            (0, reference_hue + hue_tolerance),
            (180 + reference_hue - hue_tolerance, 180),
        ]
    elif reference_hue + hue_tolerance > 180:
        # Handle the case where we wrapped around right.
        hue_ranges = [
            (reference_hue - hue_tolerance, 180),
            (0, reference_hue + hue_tolerance - 180),
        ]
    else:
        # Simplest case: just one range
        hue_ranges = [(reference_hue - hue_tolerance, reference_hue + hue_tolerance)]

    # Debugging: print the hue ranges
    # print(f"Hue ranges: {hue_ranges}")

    # Do thresholding.
    binary = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hue_lo, hue_hi in hue_ranges:
        binary |= cv.inRange(
            hsv,
            (hue_lo, saturation_lo, value_lo),  # changed from hue_hi, saturation_hi, value_hi
            (hue_hi, saturation_hi, value_hi),  # changed from hue_lo, saturation_lo, value_lo
        )

    # Apply morphological operations to clean up the binary image
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # changed k-size to 3
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    return binary


def is_shape_symmetric(
    binary: np.ndarray, centroid_xy: tuple[float, float], threshold: float, rotation: float = 0.0
) -> bool:
    """Given a binary image, return True if the shape is symmetric about its centroid when rotated
    by 'rotation' degrees, and False otherwise. Symmetry is determined by calculating intersection
    over union of (1) the original shape with (2) the rotated shape.
    """
    rot = cv.getRotationMatrix2D(
        centroid_xy, rotation, 1
    )  # changed from radian to degree for rotation
    flipped = cv.warpAffine(binary, rot, binary.shape)
    intersection = np.sum(binary & flipped)
    union = np.sum(binary ^ flipped) + intersection
    return intersection / union > threshold


def identify_single_shape(binary: np.ndarray) -> Shape:
    """Given a binary image that contains a single shape, return a string describing the shape, i.e.
    one of the SHAPE_OPTIONS shapes at the top of the file."""
    # Compute the moments of the shape and find its center of mass (AKA centroid)
    moments = cv.moments(binary)
    centroid_xy = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    # First, we can distinguish between (rectangles and wedges) vs (circles and crosses) by looking
    # at the roundedness of the shape. If the roundedness is high, it's a circle or cross. If it's
    # low, it's a rectangle or wedge.
    if roundedness(moments) < 0.5:  # changed the comparison from '>'
        # If roundedness is low, it's a rectangle or wedge. We can distinguish between these two
        # by checking if the shape is symmetric about its centroid through 180-degree rotation. If
        # it is, it's a rectangle. Otherwise, it's a wedge.
        if is_shape_symmetric(
            binary, centroid_xy, threshold=0.7, rotation=180
        ):  # changed threshold from 0.5
            return Shape.RECTANGLE
        else:
            return Shape.WEDGE
    else:
        # If roundedness is high, it's a circle or cross. Crosses are symmetric for rotations of
        # 90 degrees, but we can distinguish them from circles by checking if the shape is symmetric
        # through 45-degree rotation. If it is, it's a circle. Otherwise, it's a cross.
        if is_shape_symmetric(
            binary, centroid_xy, threshold=0.8, rotation=45
        ):  # changed threshold from 0.5
            return Shape.CIRCLE
        else:
            return Shape.CROSS


def find_shapes(
    image: np.ndarray,
    size: Size,
    color: Color,
    shape: Shape,
) -> np.ndarray:
    """Find all locations (centroids) in the image where there is a shape of the specified
    size, color, and shape type. Return the (x,y) locations of these centroids as a numpy array
    of shape (N, 2) where N is the number of shapes found.
    """
    # First pass: preprocess the image using a hue tolerance of ± 90. This gives ALL shapes that
    # are saturated, effectively ignoring the hue. We'll use this to figure out what counts as
    # 'large' or 'small' shapes, regardless of color.
    binary_all_colors = threshold_on_hue(image, color, hue_tolerance=90)

    # Second pass: threshold the image using the specified hue tolerance to get a binary image with
    # just the color we're interested in.
    binary_this_color = threshold_on_hue(image, color, hue_tolerance=HUE_TOLERANCE)

    # Use connected components to segment the binary image into individual shapes
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_all_colors)

    # Loop over the identified connected components and identify their shape, size, and color.
    # For each shape of the correct type, record its location, size, and a boolean flag indicating
    # whether it's the correct color.
    shape_info = []
    for i in range(1, num_labels):  # changed from range(num_labels)
        # Skip if shape is too small (we can't identify very very small shapes))
        if stats[i, cv.CC_STAT_AREA] < MIN_AREA:
            continue

        # Make a new temporary binary image with just the current shape
        shape_i_only = np.zeros_like(binary_all_colors)
        shape_i_only[labels == i] = 255

        # Identify the shape
        shape_type = identify_single_shape(shape_i_only)

        # If it's the kind of shape we're looking for, record its location, size, and a boolean
        # flag indicating if it is the correct color
        if shape_type == shape:
            area, is_correct_color = (
                stats[i, cv.CC_STAT_AREA],
                np.any(shape_i_only * binary_this_color),
            )
            shape_info.append((tuple(centroids[i]), area, is_correct_color))

    # Last, we need to filter out the shapes that are the wrong size. We'll do this by finding
    # a good threshold that distinguishes 'small' from 'large' for our given shape, across all
    # colors. Then we'll filter out the shapes that are the wrong size / color
    areas, counts = np.unique([area for _, area, _ in shape_info], return_counts=True)
    area_threshold = otsu_threshold(counts, bins=areas)
    if size == Size.SMALL:
        return np.array(
            [
                loc
                for loc, area, correct_color in shape_info
                if (area < area_threshold) and correct_color
            ]
        )
    else:
        return np.array(
            [
                loc
                for loc, area, correct_color in shape_info
                if (area >= area_threshold) and correct_color
            ]
        )


def annotate_locations(image: np.ndarray, locs_xy: np.ndarray) -> np.ndarray:
    """Annotate the locations on the image by drawing circles on each (x,y) location"""
    annotated = image.copy()
    black, white, symbol_size = (0, 0, 0), (255, 255, 255), 8
    for x, y in locs_xy:
        x, y = int(x), int(y)
        cv.circle(annotated, (x, y), symbol_size, white, -1, cv.LINE_AA)
        cv.line(annotated, (x - symbol_size, y), (x + symbol_size, y), black, 1, cv.LINE_AA)
        cv.line(annotated, (x, y - symbol_size), (x, y + symbol_size), black, 1, cv.LINE_AA)
        cv.circle(annotated, (x, y), symbol_size, black, 1, cv.LINE_AA)
    return annotated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "size",
        help="Return large or small shapes?",
        type=Size,
        choices=list(Size),
    )
    parser.add_argument(
        "color",
        help="Return shapes of a specific color?",
        type=Color,
        choices=list(Color),
    )
    parser.add_argument(
        "shape",
        help="Return shapes of a specific type?",
        type=Shape,
        choices=list(Shape),
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"File not found: {args.image}")

    # Load the image
    im = cv.imread(args.image)

    # Find the shapes
    locations = find_shapes(
        im,
        args.size,
        args.color,
        args.shape,
    )
    description = f"Located {len(locations)} {args.size} {args.color} {args.shape}s"

    # Annotate the locations on the image and display it
    cv.imshow(description, annotate_locations(im, locations))
    cv.waitKey(0)
    cv.destroyAllWindows()

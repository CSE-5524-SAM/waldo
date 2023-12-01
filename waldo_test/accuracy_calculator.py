import os
import cv2


def get_real_waldo(image_path, label_path):
    """
    Slice images based on labeled bounding boxes in normalized format and return a list of slices with a progress bar.

    :param image_dir: Directory containing the images
    :param label_dir: Directory containing the corresponding label files
    :return: List of sliced images containing Waldo
    """

    if os.path.exists(image_path) and os.path.exists(label_path):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
        h, w = image.shape[:2]

        # Read and process the label file
        with open(label_path, 'r') as file:
            for line in file:
                _, x_center, y_center, width, height = map(float, line.split())

                # Convert normalized coordinates to pixel coordinates
                xmin = int((x_center - width / 2) * w)
                ymin = int((y_center - height / 2) * h)
                xmax = int((x_center + width / 2) * w)
                ymax = int((y_center + height / 2) * h)

                width = (xmax - xmin) / 2
                height = (ymax - ymin) / 2


    return (xmin - width, ymin - height, xmax + width, ymax + height)


def calculate_waldo_accuracy(waldo, ind):
    if waldo is None:
        return 0

    real_waldo = get_real_waldo(f'testdata/images/test_{ind}.jpg',
                                f'testdata/real_location_label/test_{ind}.txt')

    if real_waldo:
        # Unpack the bounding box coordinates
        rxmin, rymin, rxmax, rymax = real_waldo
        wxmin, wymin, wxmax, wymax = waldo

        # Check if waldo is within real_waldo
        if wxmin >= rxmin and wymin >= rymin and wxmax <= rxmax and wymax <= rymax:
            return 1
        else:
            return 0
    else:
        print(f"No real Waldo data found for index {ind}")
        return 0

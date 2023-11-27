import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

def template_matching(test_image_gray, template):
    """
    Apply template matching for a single template image.

    :param test_image_gray: Grayscale test image where we need to find Waldo.
    :param template: One of the template images of Waldo.
    :return: List of points where Waldo was found.
    """
    # Convert the template to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    # Apply template matching
    res = cv2.matchTemplate(test_image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)

    # Return points for this template
    return [(pt[0], pt[1], w, h) for pt in zip(*loc[::-1])]


def slice_waldo_images(image_dir, label_dir):
    """
    Slice images based on labeled bounding boxes in normalized format and return a list of slices with a progress bar.

    :param image_dir: Directory containing the images
    :param label_dir: Directory containing the corresponding label files
    :return: List of sliced images containing Waldo
    """
    sliced_images = []
    image_files = os.listdir(image_dir)

    for image_filename in image_files:
        image_path = os.path.join(image_dir, image_filename)
        label_filename = image_filename.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        if os.path.exists(image_path) and os.path.exists(label_path):
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
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

                    # Slice the image
                    sliced_image = image[ymin:ymax, xmin:xmax]
                    sliced_images.append(sliced_image)

    return sliced_images

def find_waldo_with_template_matching_parallel(test_image, template_images):
    """
        Find Waldo in the test image using template matching with parallel processing.

        :param test_image: The test image where we need to find Waldo.
        :param template_images: A list of template images of Waldo.
        :return: Image with Waldo's locations marked.
        """
    # Convert the test image to grayscale
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Use ThreadPoolExecutor to run template matching in parallel
    points = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(template_matching, test_image_gray, template) for template in template_images]
        for future in futures:
            points.extend(future.result())

    # Draw rectangles for each found location
    # for pt in points:
    #     cv2.rectangle(test_image, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (0, 255, 0), 2)

    return test_image, points

def run_template_matching(test_image_path, template_paths, ind):
    test_img = cv2.imread(test_image_path)
    template_imgs = []

    for tp in template_paths:
        template_imgs.extend(slice_waldo_images(tp + '/images', tp + '/labels'))

    result_img, points = find_waldo_with_template_matching_parallel(test_img, template_imgs)
    # Create a mask with the same size as the image, initially filled with zeros
    mask = np.zeros_like(test_img, dtype=np.uint8)

    # Draw a filled rectangle on the mask for each point
    for pt in points:
        cv2.rectangle(mask, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (255, 255, 255), -1)

    # Apply Gaussian blur to the whole image
    blurred_test_img = cv2.GaussianBlur(test_img, (21, 21), 0)

    # Blend the images
    final_image = np.where(mask == np.array([255, 255, 255]), test_img, blurred_test_img)

    for pt in points:
        cv2.rectangle(final_image, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (0, 255, 0), 2)

    cv2.imwrite(f'result/template_matching/template_matching_result_{ind}.jpg', final_image, [cv2.IMWRITE_JPEG_QUALITY, 90])




import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as matches

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

def draw_final_image(waldos, i):
    image = cv2.imread(f'testdata/images/test_{i}.jpg')
    mask = np.zeros_like(image, dtype=np.uint8)
    blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
    blurred_img = np.clip(blurred_img * 0.5, 0, 255).astype(np.uint8)

    for method, w in enumerate(waldos):
        width, height = 0, 0
        if method == 3:
            width = int((w[2] - w[0]) / 2)
            height = int((w[3] - w[1]) / 2)

        if w is not None:
            cv2.rectangle(mask, (w[0] - width, w[1] - height),
                          (w[2] + width, w[3] + height), (255, 255, 255), -1)

    final_image = np.where(mask == np.array([255, 255, 255]), image, blurred_img)

    for method, w in enumerate(waldos):
        color = (0, 0, 255)
        width, height = 0, 0
        if method == 1:
            color = (0, 255, 0)
        elif method == 2:
            color = (255, 0, 0)
        elif method == 3:
            color = (255, 255, 0)
            width = int((w[2] - w[0]) / 2)
            height = int((w[3] - w[1]) / 2)

        if w is not None:
            cv2.rectangle(final_image, (w[0] - width, w[1] - height),
                          (w[2] + width, w[3] + height), color, 4)

    cv2.imwrite(f'result/result_{i}.jpg', final_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    methods = {
        "Edge Detection": (0, 255, 0),
        "Gradients": (255, 0, 0),
        "Template Matching": (0, 0, 255),
        "Neural Network": (135, 206, 235)
    }
    image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    patches = [matches.Patch(color=np.array(color) / 255, label=label)
               for label, color in methods.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=0.,
               fontsize=7, handlelength=1, handleheight=1, labelspacing=0.5)

    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def draw_result_table(result):

def show_result(results):
    g_time, ed_time, tm_time, nn_time = 0, 0, 0, 0
    g_acc, ed_acc, tm_acc, nn_acc = 0, 0, 0, 0
    for i, g_result, ed_result, tm_result, nn_result in enumerate(results):
        i = i + 1
        g_time += g_result[0]
        g_acc += calculate_waldo_accuracy(g_result[1], i)
        ed_time += ed_result[0]
        ed_acc += calculate_waldo_accuracy(ed_result[1], i)
        tm_time += tm_result[0]
        tm_acc += calculate_waldo_accuracy(tm_result[1], i)
        nn_time += nn_result[0]
        nn_acc += calculate_waldo_accuracy(nn_result[1], i)

        draw_final_image(g_result[1], i)
        draw_final_image(ed_result[1], i)
        draw_final_image(tm_result[1], i)
        draw_final_image(nn_result[1], i)




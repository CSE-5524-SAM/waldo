import os
import cv2
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as matches

def get_real_waldo(image_path, label_path):

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

def draw_final_image(result, i, ax):
    waldos = [result[0][1], result[1][1], result[2][1], result[3][1]]

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
    ax.imshow(image_rgb)
    ax.axis('off')
    ax.legend(handles=patches, bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=0.,
              fontsize=7, handlelength=1, handleheight=1, labelspacing=0.5)

def draw_result_table(results):
    g_time, ed_time, tm_time, nn_time = 0, 0, 0, 0
    g_acc, ed_acc, tm_acc, nn_acc = 0, 0, 0, 0
    total = len(results)
    for i, result in enumerate(results):
        i = i + 1
        g_result, ed_result, tm_result, nn_result = result[0], result[1], result[2], result[3]

        g_time += round(g_result[0], 2)
        g_acc += calculate_waldo_accuracy(g_result[1], i)
        ed_time += round(ed_result[0], 2)
        ed_acc += calculate_waldo_accuracy(ed_result[1], i)
        tm_time += round(tm_result[0],2)
        tm_acc += calculate_waldo_accuracy(tm_result[1], i)
        nn_time += round(nn_result[0],2 )
        nn_acc += calculate_waldo_accuracy(nn_result[1], i)

    data = {
        "Methods": ["Gradients", "Edge Detection", "Template Matching", "Neural Network"],
        "Accuracy": [g_acc/total, ed_acc/total, tm_acc/total, nn_acc/total],
        "Time (without multithreading)": [g_time/total, ed_time/total, '-', nn_time/total],
        "Time (with multithreading)": ['-', '-', tm_time/total, '-']
    }

    df = pd.DataFrame(data)
    # Set the 'Methods' column as index if needed
    df.set_index('Methods', inplace=True)
    df.style.set_properties(**{'text-align': 'left'})
    display(df)


def show_result(results):
    num_images = len(results)
    cols = 2
    rows = (num_images + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 5))

    draw_result_table(results)
    for i, result in enumerate(results):
        if rows > 1:
            ax = axes[i // cols, i % cols]
        else:
            ax = axes[i % cols]
        draw_final_image(result, i + 1, ax)

    plt.tight_layout()
    plt.show()





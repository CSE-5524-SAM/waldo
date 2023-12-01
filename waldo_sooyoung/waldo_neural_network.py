import cv2
from ultralytics import YOLO


def run_neural_network(ind):
    model = YOLO('waldo_ml.pt')
    pred_result = model(f'testdata/images/test_{ind}.jpg', verbose=False)
    bbox = []
    for r in pred_result:
        bbox = r.boxes.xyxy[0]

    image = cv2.imread(f'testdata/images/test_{ind}.jpg')
    # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
    blur = cv2.GaussianBlur(image, (31, 31), 0)
    min_x = int(bbox[0])
    min_y = int(bbox[1])
    max_x = int(bbox[2])
    max_y = int(bbox[3])
    result = (min_x, min_y, max_x, max_y)
    xl = (max_x - min_x) // 2
    yl = (max_y - min_y) // 2
    min_x = min_x - xl
    min_y = min_y - yl
    max_x = max_x + xl
    max_y = max_y + yl

    blur[min_y:max_y, min_x:max_x] = image[min_y:max_y, min_x:max_x]
    cv2.rectangle(blur, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    cv2.imwrite(f'result/ml/ml_result_{ind}.jpg', blur)

    return result
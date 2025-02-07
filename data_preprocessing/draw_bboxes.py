import cv2
import os
import matplotlib.pyplot as plt


def draw_yolo_bboxes(image_path, label_path, output_path=None, img_show=False):
    # load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    height, width, _ = image.shape

    # load labels
    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

        # convert YOLO format to absolute coordinates
        x_center_abs = int(x_center * width)
        y_center_abs = int(y_center * height)
        bbox_width_abs = int(bbox_width * width)
        bbox_height_abs = int(bbox_height * height)

        # calculate the top-left corner
        x1 = int(x_center_abs - bbox_width_abs / 2)
        y1 = int(y_center_abs - bbox_height_abs / 2)
        x2 = int(x_center_abs + bbox_width_abs / 2)
        y2 = int(y_center_abs + bbox_height_abs / 2)

        # draw bbox
        color = (0, 255, 0)  # green
        thickness = 2  # line thickness
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # add class label
        text = f"Class {class_id}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    if img_show:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, os.path.basename(image_path))
        cv2.imwrite(output_file, image)

        print(f"Image saved with BBoxes: {output_file}")

    return image


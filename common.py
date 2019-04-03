"""
Common logic across inference on Inference Engine and TensorFlow
"""

import cv2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def read_resize_image(path_to_image, width, height):
    """
    Takes and image and resizes it to the given dimensions
    """
    raw_image = cv2.imread(path_to_image)
    return cv2.resize(raw_image, (width, height), interpolation=cv2.INTER_NEAREST)


# pylint: disable=too-many-locals
def draw_image(original_image, res, path_to_image, prob_threshold=0.8, color=(0, 255, 0)):
    """
    Takes a path to the image and bounding boxes. Draws those boxes on the new image and saves it
    """
    raw_image = cv2.imread(original_image)
    initial_w = raw_image.shape[1]
    initial_h = raw_image.shape[0]
    labels_map = {
        21: 'cat'
    }
    for obj in res[0][0]:
        # Draw only objects when probability more than specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            class_id = int(obj[1])
            cv2.rectangle(raw_image, (xmin, ymin), (xmax, ymax), color, 2)
            det_label = labels_map[class_id] if labels_map else str(class_id)
            cv2.putText(raw_image,
                        det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    cv2.imwrite(path_to_image, raw_image)


def show_results_interactively(tf_image, ie_image, combination_image, ie_fps, tf_fps):
    """
    Takes paths to three images and shows them with matplotlib on one screen
    """
    _ = plt.figure(figsize=(30, 10))
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(wspace=0.25, hspace=0.05)

    titles = [
        '(a) Tensorflow',
        '(b) Inference Engine',
        '(c) TensorFlow and Inference Engine\n predictions are identical'
    ]

    for i, path in enumerate([tf_image, ie_image, combination_image]):
        img_resized = cv2.imread(path)
        ax_plot = plt.subplot(gs1[i])
        ax_plot.axis("off")
        addon = ' '
        if i == 1:
            addon += '{:4.3f}'.format(ie_fps) + '(FPS)'
        elif i == 0:
            addon += '{:4.3f}'.format(tf_fps) + '(FPS)'

        ax_plot.text(0.5, -0.5, titles[i] + addon,
                     size=12, ha="center",
                     transform=ax_plot.transAxes)
        ax_plot.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

    plt.show()

import argparse
from math import ceil
from itertools import product

import cv2
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

INPUT_SIZE = [608, 640]


class PriorBox(object):
    def __init__(
        self,
        image_size: list,
        min_sizes: list = [[16, 32], [64, 128], [256, 512]],
        steps: list = [8, 16, 32],
        clip: bool = False,
    ):
        super(PriorBox, self).__init__()
        self.image_size = image_size
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def forward(self) -> np.ndarray:
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [
                        x * self.steps[k] / self.image_size[1] for x in [j + 0.5]
                    ]
                    dense_cy = [
                        y * self.steps[k] / self.image_size[0] for y in [i + 0.5]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = np.array(anchors).reshape(-1, 4)

        return output


def pad_image(
    image: np.ndarray, h: int, w: int, size: list, padvalue: float
) -> np.ndarray:
    pad_image = image.copy()
    pad_h = max(size[0] - h, 0)
    pad_w = max(size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue
        )
    return pad_image


def resize_image(
    image: np.ndarray, re_size: list, keep_ratio: bool = True
) -> tuple[: np.ndarray, float]:
    if not keep_ratio:
        re_image = cv2.resize(image, (re_size[0], re_size[1])).astype("float32")
        return re_image, 0, 0
    ratio = re_size[0] * 1.0 / re_size[1]
    h, w = image.shape[0:2]
    if h * 1.0 / w <= ratio:
        resize_ratio = re_size[1] * 1.0 / w
        re_h, re_w = int(h * resize_ratio), re_size[1]
    else:
        resize_ratio = re_size[0] * 1.0 / h
        re_h, re_w = re_size[0], int(w * resize_ratio)

    re_image = cv2.resize(image, (re_w, re_h)).astype("float32")
    re_image = pad_image(re_image, re_h, re_w, re_size, (0.0, 0.0, 0.0))
    return re_image, resize_ratio


def preprocess(
    img_raw: np.ndarray, input_size: list
) -> tuple[np.ndarray, np.ndarray, float]:
    img = np.float32(img_raw)
    img, resize = resize_image(img, input_size)
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img, scale, resize


def decode(
    loc: np.ndarray, priors: np.ndarray, variances: list = [0.1, 0.2]
) -> np.ndarray:
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def decode_landm(
    pre: np.ndarray, priors: np.ndarray, variances: list = [0.1, 0.2]
) -> np.ndarray:
    landms = np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        1,
    )
    return landms


def py_cpu_nms(dets: np.ndarray, thresh: float) -> list:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def infer(
    image_path: str, confidence_threshold: float = 0.4, nms_threshold: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img, scale, resize = preprocess(img_raw, INPUT_SIZE)
    img = np.transpose(img, (0, 2, 3, 1))
    run_ort = ort.InferenceSession("./weights/RetinaFace_int.onnx")
    outputs = run_ort.run(None, {run_ort.get_inputs()[0].name: img})
    _, im_height, im_width, _ = img.shape
    loc = outputs[0]
    conf = outputs[1]
    landms = outputs[2]

    conf = softmax(conf, axis=-1)

    priorbox = PriorBox(image_size=(im_height, im_width))
    priors = priorbox.forward()
    boxes = decode(loc.squeeze(0), priors)
    boxes = boxes * scale / resize
    scores = conf.squeeze(0)[:, 1]
    landms = decode_landm(landms.squeeze(0), priors)
    scale1 = np.array(
        [
            img.shape[2],
            img.shape[1],
            img.shape[2],
            img.shape[1],
            img.shape[2],
            img.shape[1],
            img.shape[2],
            img.shape[1],
            img.shape[2],
            img.shape[1],
        ]
    )
    landms = landms * scale1 / resize

    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    dets = np.concatenate((dets, landms), axis=1)
    return dets, img_raw


def is_square(img_path: str) -> bool:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im_height, im_width, _ = img.shape
    if im_height == im_width:
        return True
    return False


def crop_to_square(
    img_path: str, output_path: str = None, save_to_file: bool = False
) -> np.ndarray:
    if is_square(img_path):
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    result, img = infer(img_path)
    x1, y1, x2, y2 = result[0][:4].astype(int).tolist()
    im_height, im_width, _ = img.shape
    square_size = min(im_height, im_width)
    vertical = True if (im_height > im_width) else False
    if vertical:
        face_cropped_image = img[y1:y2, :]
        crop_f_height, crop_f_width, _ = face_cropped_image.shape
        complement = square_size - crop_f_height
        half_comp = complement // 2
        cropped_image = img[
            y1 - half_comp : y2 + half_comp,
            :,
        ]
        if save_to_file:
            cv2.imwrite(output_path, cropped_image)
        return cropped_image

    if not vertical:
        face_cropped_image = img[:, x1:x2]
        crop_f_height, crop_f_width, _ = face_cropped_image.shape
        complement = square_size - crop_f_width
        half_comp = complement // 2
        cropped_image = img[
            :,
            x1 - half_comp : x2 + half_comp,
        ]
        if save_to_file:
            cv2.imwrite(output_path, cropped_image)
        return cropped_image


def main(img_path: str, output_path: str):
    crop_to_square(img_path=img_path, output_path=output_path, save_to_file=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceCropper")
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-o", "--output_path")
    args = parser.parse_args()
    main(img_path=args.input_path, output_path=args.output_path)

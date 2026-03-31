import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

# (x, y, w, h, confidence, label)
Detection = Tuple[int, int, int, int, float, str]


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class YOLOv8Detector(BaseDetector):
    def __init__(self, model_size: str = "n", conf_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.available = False
        self._model_size = model_size
        try:
            from ultralytics import YOLO
            self._model = YOLO(f"yolov8{model_size}.pt")
            self.available = True
        except Exception as e:
            print(f"[YOLOv8] Not available: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.available:
            return []
        try:
            results = self._model(frame, conf=self.conf_threshold, verbose=False)
            detections: List[Detection] = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self._model.names[cls]
                    detections.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), conf, label))
            return detections
        except Exception as e:
            print(f"[YOLOv8] Inference error: {e}")
            return []

    def get_name(self) -> str:
        return f"YOLOv8-{self._model_size.upper()}"


class HOGDetector(BaseDetector):
    def __init__(self):
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        try:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            rects, weights = self._hog.detectMultiScale(
                gray, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            detections: List[Detection] = []
            for i, (x, y, w, h) in enumerate(rects):
                conf = float(weights[i][0]) if len(weights) > i else 0.7
                detections.append((x * 2, y * 2, w * 2, h * 2, conf, "person"))
            return detections
        except Exception:
            return []

    def get_name(self) -> str:
        return "HOG + SVM"


class BackgroundSubtractorDetector(BaseDetector):
    def __init__(self, method: str = "MOG2", min_area: int = 600):
        self._method = method
        self._min_area = min_area
        if method == "MOG2":
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=50, detectShadows=True
            )
        else:
            self._subtractor = cv2.createBackgroundSubtractorKNN(
                history=200, dist2Threshold=400, detectShadows=True
            )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        try:
            mask = self._subtractor.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
            mask = cv2.dilate(mask, self._kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections: List[Detection] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= self._min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    conf = min(area / 8000.0, 1.0)
                    detections.append((x, y, w, h, conf, "motion"))
            return detections
        except Exception:
            return []

    def get_name(self) -> str:
        return f"Background Subtraction ({self._method})"


class HaarCascadeDetector(BaseDetector):
    CASCADE_FILES = {
        "face": "haarcascade_frontalface_default.xml",
        "body": "haarcascade_fullbody.xml",
        "upper_body": "haarcascade_upperbody.xml",
        "profile_face": "haarcascade_profileface.xml",
    }

    def __init__(self, cascade_type: str = "face"):
        self._cascade_type = cascade_type
        filename = self.CASCADE_FILES.get(cascade_type, "haarcascade_frontalface_default.xml")
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
        self._label = cascade_type.replace("_", " ")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            objects = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            detections: List[Detection] = []
            for x, y, w, h in objects:
                detections.append((int(x), int(y), int(w), int(h), 0.8, self._label))
            return detections
        except Exception:
            return []

    def get_name(self) -> str:
        return f"Haar Cascade ({self._cascade_type.replace('_', ' ').title()})"


class OpticalFlowDetector(BaseDetector):
    """Dense optical flow (Farneback) based motion detector."""

    def __init__(self, threshold: float = 2.0, min_area: int = 500):
        self._prev_gray = None
        self._threshold = threshold
        self._min_area = min_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            self._prev_gray = gray
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask = (magnitude > self._threshold).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.dilate(mask, self._kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections: List[Detection] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= self._min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    conf = min(area / 6000.0, 1.0)
                    detections.append((x, y, w, h, conf, "flow"))
            return detections
        except Exception:
            return []

    def get_name(self) -> str:
        return "Optical Flow (Farneback)"


DETECTOR_REGISTRY = {
    "Background Subtraction (MOG2)": lambda: BackgroundSubtractorDetector("MOG2"),
    "Background Subtraction (KNN)": lambda: BackgroundSubtractorDetector("KNN"),
    "Optical Flow (Farneback)": lambda: OpticalFlowDetector(),
    "HOG + SVM (Pedestrian)": lambda: HOGDetector(),
    "Haar Cascade (Face)": lambda: HaarCascadeDetector("face"),
    "Haar Cascade (Body)": lambda: HaarCascadeDetector("body"),
    "Haar Cascade (Upper Body)": lambda: HaarCascadeDetector("upper_body"),
    "YOLOv8-Nano": lambda: YOLOv8Detector("n"),
    "YOLOv8-Small": lambda: YOLOv8Detector("s"),
    "YOLOv8-Medium": lambda: YOLOv8Detector("m"),
}

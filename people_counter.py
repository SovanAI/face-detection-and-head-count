import cv2
import numpy as np
import threading
import time
import os


class PeopleCounter:

    def __init__(self):

        # Webcam
        camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_index}. "
                "Set CAMERA_INDEX to a valid camera device."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )

        # ------------------------------------------------
        # Gate line
        #
        # "vertical" means:
        #
        #        │
        #        │ gate
        #        │
        #
        # People move LEFT <-> RIGHT
        # ------------------------------------------------

        self.line_orientation = "vertical"

        # Percentage of frame where gate line is placed
        self.line_position = 0.5

        # Tracking data
        self.objects = {}

        self.next_object_id = 0

        # Counters
        self.entry_count = 0
        self.exit_count = 0

        # First detected direction
        self.entry_direction = None
        self.exit_direction = None

        # Lock for FastAPI
        self.lock = threading.Lock()

        # Latest frame
        self.latest_frame = None

        self.running = True

        # Start camera processing
        self.thread = threading.Thread(
            target=self.process_camera,
            daemon=True
        )

        self.thread.start()

    # ------------------------------------------------
    # Calculate centroid
    # ------------------------------------------------

    def get_centroid(self, x, y, w, h):

        cx = int(x + w / 2)
        cy = int(y + h / 2)

        return cx, cy

    # ------------------------------------------------
    # Distance between two points
    # ------------------------------------------------

    def distance(self, p1, p2):

        return np.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    # ------------------------------------------------
    # Detect moving objects
    # ------------------------------------------------

    def detect_people(self, frame):

        mask = self.bg_subtractor.apply(frame)

        # Remove shadows
        _, mask = cv2.threshold(
            mask,
            200,
            255,
            cv2.THRESH_BINARY
        )

        # Remove noise
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore small objects
            if area < 1500:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Ignore very small detections
            if w < 30 or h < 50:
                continue

            cx, cy = self.get_centroid(
                x, y, w, h
            )

            detections.append({
                "bbox": (x, y, w, h),
                "centroid": (cx, cy)
            })

        return detections

    # ------------------------------------------------
    # Match detections to existing objects
    # ------------------------------------------------

    def update_tracking(self, detections):

        max_distance = 100

        matched_objects = set()

        for detection in detections:

            centroid = detection["centroid"]

            best_id = None
            best_distance = max_distance

            for object_id, obj in self.objects.items():

                if object_id in matched_objects:
                    continue

                distance = self.distance(
                    centroid,
                    obj["centroid"]
                )

                if distance < best_distance:

                    best_distance = distance
                    best_id = object_id

            # Existing object
            if best_id is not None:

                old_centroid = self.objects[
                    best_id
                ]["centroid"]

                self.objects[best_id]["previous_centroid"] = (
                    old_centroid
                )

                self.objects[best_id]["centroid"] = centroid

                self.objects[best_id]["bbox"] = detection["bbox"]

                self.objects[best_id]["last_seen"] = time.time()

                matched_objects.add(best_id)

            # New object
            else:

                object_id = self.next_object_id

                self.next_object_id += 1

                self.objects[object_id] = {

                    "centroid": centroid,

                    "previous_centroid": centroid,

                    "bbox": detection["bbox"],

                    "last_seen": time.time(),

                    "counted": False
                }

                matched_objects.add(object_id)

    # ------------------------------------------------
    # Check direction and crossing
    # ------------------------------------------------

    def check_crossings(self):

        # Calculate actual line position
        if self.latest_frame is None:
            return

        height, width = self.latest_frame.shape[:2]

        if self.line_orientation == "vertical":

            line = int(width * self.line_position)

        else:

            line = int(height * self.line_position)

        for object_id, obj in list(self.objects.items()):

            previous = obj["previous_centroid"]
            current = obj["centroid"]

            # ----------------------------------------
            # Vertical line
            # LEFT <-> RIGHT
            # ----------------------------------------

            if self.line_orientation == "vertical":

                previous_x = previous[0]
                current_x = current[0]

                # Moving LEFT -> RIGHT
                if (
                    previous_x < line
                    and current_x >= line
                ):

                    direction = "left_to_right"

                    self.register_crossing(
                        object_id,
                        direction
                    )

                # Moving RIGHT -> LEFT
                elif (
                    previous_x > line
                    and current_x <= line
                ):

                    direction = "right_to_left"

                    self.register_crossing(
                        object_id,
                        direction
                    )

            # ----------------------------------------
            # Horizontal line
            # TOP <-> BOTTOM
            # ----------------------------------------

            else:

                previous_y = previous[1]
                current_y = current[1]

                # TOP -> BOTTOM
                if (
                    previous_y < line
                    and current_y >= line
                ):

                    direction = "top_to_bottom"

                    self.register_crossing(
                        object_id,
                        direction
                    )

                # BOTTOM -> TOP
                elif (
                    previous_y > line
                    and current_y <= line
                ):

                    direction = "bottom_to_top"

                    self.register_crossing(
                        object_id,
                        direction
                    )

    # ------------------------------------------------
    # Register crossing
    # ------------------------------------------------

    def register_crossing(
        self,
        object_id,
        direction
    ):

        obj = self.objects[object_id]

        # Don't count same person twice
        if obj["counted"]:
            return

        # --------------------------------------------
        # FIRST PERSON
        # --------------------------------------------

        if self.entry_direction is None:

            self.entry_direction = direction

            self.exit_direction = self.get_opposite_direction(
                direction
            )

            self.entry_count += 1

            obj["counted"] = True

            print(
                f"Entry direction established: {direction}"
            )

            return

        # --------------------------------------------
        # ENTRY
        # --------------------------------------------

        if direction == self.entry_direction:

            self.entry_count += 1

            obj["counted"] = True

        # --------------------------------------------
        # EXIT
        # --------------------------------------------

        elif direction == self.exit_direction:

            self.exit_count += 1

            obj["counted"] = True

    # ------------------------------------------------
    # Opposite direction
    # ------------------------------------------------

    def get_opposite_direction(self, direction):

        opposites = {

            "left_to_right": "right_to_left",

            "right_to_left": "left_to_right",

            "top_to_bottom": "bottom_to_top",

            "bottom_to_top": "top_to_bottom"
        }

        return opposites[direction]

    # ------------------------------------------------
    # Draw information
    # ------------------------------------------------

    def draw_ui(self, frame):

        height, width = frame.shape[:2]

        if self.line_orientation == "vertical":

            line_x = int(width * self.line_position)

            cv2.line(
                frame,
                (line_x, 0),
                (line_x, height),
                (0, 255, 255),
                3
            )

        else:

            line_y = int(height * self.line_position)

            cv2.line(
                frame,
                (0, line_y),
                (width, line_y),
                (0, 255, 255),
                3
            )

        # Draw tracked objects
        for object_id, obj in self.objects.items():

            x, y, w, h = obj["bbox"]

            cx, cy = obj["centroid"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"ID: {object_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Display counters

        cv2.putText(
            frame,
            f"ENTRY: {self.entry_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"EXIT: {self.exit_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if self.entry_direction:

            cv2.putText(
                frame,
                f"Entry Direction: {self.entry_direction}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        return frame

    # ------------------------------------------------
    # Camera loop
    # ------------------------------------------------

    def process_camera(self):

        while self.running:

            success, frame = self.cap.read()

            if not success:
                continue

            detections = self.detect_people(frame)

            self.update_tracking(detections)

            self.latest_frame = frame

            self.check_crossings()

            frame = self.draw_ui(frame)

            self.latest_frame = frame

    # ------------------------------------------------
    # Get latest frame
    # ------------------------------------------------

    def get_frame(self):

        with self.lock:

            if self.latest_frame is None:
                return None

            return self.latest_frame.copy()

    # ------------------------------------------------
    # Get statistics
    # ------------------------------------------------

    def get_stats(self):

        with self.lock:

            return {
                "entry_count": self.entry_count,

                "exit_count": self.exit_count,

                "entry_direction": self.entry_direction,

                "exit_direction": self.exit_direction,

                "current_inside": (
                    self.entry_count
                    - self.exit_count
                )
            }

    # ------------------------------------------------
    # Reset
    # ------------------------------------------------

    def reset(self):

        with self.lock:

            self.entry_count = 0
            self.exit_count = 0

            self.entry_direction = None
            self.exit_direction = None

            self.objects = {}

            self.next_object_id = 0

    # ------------------------------------------------
    # Release
    # ------------------------------------------------

    def release(self):

        self.running = False

        self.cap.release()
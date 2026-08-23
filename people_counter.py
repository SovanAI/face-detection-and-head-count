import cv2
import numpy as np
import threading
import time
import os

class PeopleCounter:
    def __init__(self):
        # Webcam configuration
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

        # Initialize HOG descriptor for human detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Gate line configuration
        self.line_orientation = "vertical"
        self.line_position = 0.5

        # Detection configurations
        self.detection_mode = "hybrid"  # Options: "motion", "hog", "hybrid"
        self.max_disappeared = 12       # Frame budget before track deletion

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

        # Start camera processing thread
        self.thread = threading.Thread(
            target=self.process_camera,
            daemon=True
        )
        self.thread.start()

    def get_centroid(self, x, y, w, h):
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        return cx, cy

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # ------------------------------------------------
    # Detection Implementations
    # ------------------------------------------------

    def detect_people_motion(self, frame):
        mask = self.bg_subtractor.apply(frame)

        # Remove shadows
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

            cx, cy = self.get_centroid(x, y, w, h)
            detections.append({
                "bbox": (x, y, w, h),
                "centroid": (cx, cy)
            })

        return detections

    def detect_people_hog(self, frame):
        height, width = frame.shape[:2]
        
        # Downscale for faster detection performance on CPU
        target_w = 640
        target_h = int(height * (target_w / width))
        
        scale_x = width / target_w
        scale_y = height / target_h
        
        small_frame = cv2.resize(frame, (target_w, target_h))
        
        # Run HOG multi-scale detector
        boxes, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
        
        detections = []
        if len(boxes) > 0:
            # Apply Non-Maximum Suppression to filter overlapping detections
            bbox_list = [[int(bx), int(by), int(bw), int(bh)] for (bx, by, bw, bh) in boxes]
            confidences = [float(w) for w in weights]
            indices = cv2.dnn.NMSBoxes(bbox_list, confidences, score_threshold=0.2, nms_threshold=0.4)
            
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = bbox_list[idx]
                    
                    orig_x = int(bx * scale_x)
                    orig_y = int(by * scale_y)
                    orig_w = int(bw * scale_x)
                    orig_h = int(bh * scale_y)
                    
                    cx, cy = self.get_centroid(orig_x, orig_y, orig_w, orig_h)
                    detections.append({
                        "bbox": (orig_x, orig_y, orig_w, orig_h),
                        "centroid": (cx, cy)
                    })
                    
        return detections

    def detect_people_hybrid(self, frame):
        motion_detections = self.detect_people_motion(frame)
        if not motion_detections:
            return []
            
        detections = []
        height, width = frame.shape[:2]
        
        for det in motion_detections:
            x, y, w, h = det["bbox"]
            
            # Crop ROI with 20% padding
            pad_w = int(w * 0.2)
            pad_h = int(h * 0.2)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(width, x + w + pad_w)
            y2 = min(height, y + h + pad_h)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            # Ensure ROI is at least the minimum HOG detection window size (64x128) to prevent assertion failures
            if roi.shape[1] < 64 or roi.shape[0] < 128:
                roi = cv2.resize(roi, (64, 128))
                
            # Run HOG detector on moving ROI (extremely fast since resolution is tiny)
            boxes, weights = self.hog.detectMultiScale(
                roi,
                winStride=(4, 4),
                padding=(4, 4),
                scale=1.05
            )
            
            # If a person is confirmed with reasonable weight
            if len(boxes) > 0 and max(weights) > 0.25:
                detections.append(det)
                
        return detections

    def detect_people(self, frame):
        if self.detection_mode == "motion":
            return self.detect_people_motion(frame)
        elif self.detection_mode == "hog":
            return self.detect_people_hog(frame)
        elif self.detection_mode == "hybrid":
            return self.detect_people_hybrid(frame)
        return self.detect_people_motion(frame)

    # ------------------------------------------------
    # Tracking logic
    # ------------------------------------------------

    def update_tracking(self, detections):
        max_distance = 120  # Max distance to consider a match

        # 1. Increment disappeared counter for all active objects
        for obj_id in self.objects:
            self.objects[obj_id]["disappeared"] += 1

        matched_objects = set()
        matched_detections = set()

        # If there are active tracks and new detections, match them
        if len(self.objects) > 0 and len(detections) > 0:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid]["centroid"] for oid in object_ids])
            detection_centroids = np.array([det["centroid"] for det in detections])

            # Pairwise matching
            for d_idx, det_centroid in enumerate(detection_centroids):
                distances = np.linalg.norm(object_centroids - det_centroid, axis=1)
                sorted_indices = np.argsort(distances)
                
                for o_idx in sorted_indices:
                    dist = distances[o_idx]
                    obj_id = object_ids[o_idx]

                    if dist > max_distance:
                        break

                    if obj_id in matched_objects:
                        continue

                    # Update matched track
                    old_centroid = self.objects[obj_id]["centroid"]
                    self.objects[obj_id]["previous_centroid"] = old_centroid
                    self.objects[obj_id]["centroid"] = tuple(det_centroid)
                    self.objects[obj_id]["bbox"] = detections[d_idx]["bbox"]
                    self.objects[obj_id]["last_seen"] = time.time()
                    self.objects[obj_id]["disappeared"] = 0
                    
                    # Track history
                    self.objects[obj_id]["history"].append(tuple(det_centroid))
                    if len(self.objects[obj_id]["history"]) > 20:
                        self.objects[obj_id]["history"].pop(0)

                    matched_objects.add(obj_id)
                    matched_detections.add(d_idx)
                    break

        # 2. For unmatched detections, spawn new tracks
        for d_idx, detection in enumerate(detections):
            if d_idx in matched_detections:
                continue

            centroid = detection["centroid"]
            object_id = self.next_object_id
            self.next_object_id += 1

            # Generate random bright BGR color
            color = (
                int(np.random.randint(80, 255)),
                int(np.random.randint(80, 255)),
                int(np.random.randint(80, 255))
            )

            self.objects[object_id] = {
                "centroid": centroid,
                "previous_centroid": centroid,
                "bbox": detection["bbox"],
                "last_seen": time.time(),
                "counted": False,
                "disappeared": 0,
                "history": [centroid],
                "color": color
            }

        # 3. Clean up expired tracks
        for obj_id in list(self.objects.keys()):
            if self.objects[obj_id]["disappeared"] > self.max_disappeared:
                del self.objects[obj_id]

    # ------------------------------------------------
    # Crossings and gate logic
    # ------------------------------------------------

    def check_crossings(self):
        if self.latest_frame is None:
            return

        height, width = self.latest_frame.shape[:2]

        if self.line_orientation == "vertical":
            line = int(width * self.line_position)
        else:
            line = int(height * self.line_position)
            # Ensure line position stays below header banner (70px)
            line = max(85, line)

        for object_id, obj in list(self.objects.items()):
            # Only count if the track was updated on this frame
            if obj["disappeared"] > 0:
                continue

            previous = obj["previous_centroid"]
            current = obj["centroid"]

            # ----------------------------------------
            # Vertical line (LEFT <-> RIGHT)
            # ----------------------------------------
            if self.line_orientation == "vertical":
                previous_x = previous[0]
                current_x = current[0]

                # Moving LEFT -> RIGHT
                if previous_x < line and current_x >= line:
                    self.register_crossing(object_id, "left_to_right")
                # Moving RIGHT -> LEFT
                elif previous_x > line and current_x <= line:
                    self.register_crossing(object_id, "right_to_left")

            # ----------------------------------------
            # Horizontal line (TOP <-> BOTTOM)
            # ----------------------------------------
            else:
                previous_y = previous[1]
                current_y = current[1]

                # Moving TOP -> BOTTOM
                if previous_y < line and current_y >= line:
                    self.register_crossing(object_id, "top_to_bottom")
                # Moving BOTTOM -> TOP
                elif previous_y > line and current_y <= line:
                    self.register_crossing(object_id, "bottom_to_top")

    def register_crossing(self, object_id, direction):
        obj = self.objects[object_id]

        if obj["counted"]:
            return

        # Setup directional standard based on first crossing
        if self.entry_direction is None:
            self.entry_direction = direction
            self.exit_direction = self.get_opposite_direction(direction)
            self.entry_count += 1
            obj["counted"] = True
            print(f"Entry direction established: {direction}")
            return

        # Increment count based on matching direction
        if direction == self.entry_direction:
            self.entry_count += 1
            obj["counted"] = True
        elif direction == self.exit_direction:
            self.exit_count += 1
            obj["counted"] = True

    def get_opposite_direction(self, direction):
        opposites = {
            "left_to_right": "right_to_left",
            "right_to_left": "left_to_right",
            "top_to_bottom": "bottom_to_top",
            "bottom_to_top": "top_to_bottom"
        }
        return opposites[direction]

    # ------------------------------------------------
    # UI HUD drawing
    # ------------------------------------------------

    def draw_corner_rect(self, img, bbox, color, thickness=2, length=15):
        x, y, w, h = bbox
        
        # Clip coordinates to prevent drawing outside image boundaries
        height, width = img.shape[:2]
        x, y = max(0, x), max(0, y)
        w = min(width - x, w)
        h = min(height - y, h)

        # Top-left corner
        cv2.line(img, (x, y), (x + length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + length), color, thickness)
        # Top-right corner
        cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
        # Bottom-left corner
        cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
        # Bottom-right corner
        cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

    def draw_ui(self, frame):
        height, width = frame.shape[:2]
        
        # 1. Overlay a semi-transparent HUD top banner (Slate-Blue background)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 70), (30, 20, 15), -1)  # BGR slate color
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # 2. Draw blinking active dot and live state
        dot_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (100, 100, 100)
        cv2.circle(frame, (25, 35), 6, dot_color, -1)
        cv2.putText(frame, "SYS LIVE", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        mode_str = f"MODE: {self.detection_mode.upper()}"
        cv2.putText(frame, mode_str, (135, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # 3. Draw statistics counters (rounded boxes)
        # ENTRANCE (Green)
        cv2.rectangle(frame, (width - 480, 15), (width - 340, 55), (30, 45, 30), -1)
        cv2.rectangle(frame, (width - 480, 15), (width - 340, 55), (0, 255, 0), 1)
        cv2.putText(frame, "IN", (width - 470, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.entry_count}", (width - 470, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        # EXIT (Red)
        cv2.rectangle(frame, (width - 320, 15), (width - 180, 55), (30, 30, 45), -1)
        cv2.rectangle(frame, (width - 320, 15), (width - 180, 55), (0, 0, 255), 1)
        cv2.putText(frame, "OUT", (width - 310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.exit_count}", (width - 310, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        # OCCUPANCY (Cyan)
        current_inside = max(0, self.entry_count - self.exit_count)
        cv2.rectangle(frame, (width - 160, 15), (width - 20, 55), (45, 45, 30), -1)
        cv2.rectangle(frame, (width - 160, 15), (width - 20, 55), (255, 255, 0), 1)
        cv2.putText(frame, "INSIDE", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{current_inside}", (width - 150, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Draw Gate Line
        if self.line_orientation == "vertical":
            line_x = int(width * self.line_position)
            cv2.line(frame, (line_x, 70), (line_x, height), (0, 180, 255), 2)
            cv2.line(frame, (line_x - 1, 70), (line_x - 1, height), (0, 255, 255), 1)
            cv2.line(frame, (line_x + 1, 70), (line_x + 1, height), (0, 255, 255), 1)
            cv2.putText(frame, "< ENTRY", (line_x - 70, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "EXIT >", (line_x + 15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            line_y = int(height * self.line_position)
            line_y = max(85, line_y)  # ensure below HUD
            cv2.line(frame, (0, line_y), (width, line_y), (0, 180, 255), 2)
            cv2.line(frame, (0, line_y - 1), (width, line_y - 1), (0, 255, 255), 1)
            cv2.line(frame, (0, line_y + 1), (width, line_y + 1), (0, 255, 255), 1)
            cv2.putText(frame, "^ ENTRY", (20, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "v EXIT", (20, line_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # 5. Draw tracked targets, trails, and corner boxes
        for object_id, obj in self.objects.items():
            if obj["disappeared"] > 2:
                continue

            x, y, w, h = obj["bbox"]
            cx, cy = obj["centroid"]
            color = obj.get("color", (0, 255, 0))

            # Draw target corners
            self.draw_corner_rect(frame, (x, y, w, h), color, thickness=2, length=15)
            
            # Semi-transparent target body
            try:
                sub_rect = frame[y:y+h, x:x+w]
                if sub_rect.size > 0:
                    rect_overlay = sub_rect.copy()
                    cv2.rectangle(rect_overlay, (0, 0), (w, h), color, -1)
                    cv2.addWeighted(rect_overlay, 0.1, sub_rect, 0.9, 0, sub_rect)
                    frame[y:y+h, x:x+w] = sub_rect
            except Exception:
                pass

            # Target locked indicator
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.circle(frame, (cx, cy), 1, (255, 255, 255), -1)

            # Draw path/trail history
            history = obj.get("history", [])
            if len(history) > 1:
                pts = np.array(history, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)
                for pt in history[:-1]:
                    cv2.circle(frame, pt, 2, color, -1)

            # Floating tag label
            tag_text = f"ID: {object_id}"
            (tw, th), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            tag_y = max(th + 10, y - 8)
            cv2.rectangle(frame, (x, tag_y - th - 4), (x + tw + 8, tag_y + baseline - 2), color, -1)
            cv2.putText(frame, tag_text, (x + 4, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------
    # Camera thread loop
    # ------------------------------------------------

    def process_camera(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                # brief pause if frame grab failed
                time.sleep(0.01)
                continue

            with self.lock:
                detections = self.detect_people(frame)
                self.update_tracking(detections)
                self.latest_frame = frame
                self.check_crossings()
                frame = self.draw_ui(frame)
                self.latest_frame = frame

            # Yield CPU slice
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_stats(self):
        with self.lock:
            return {
                "entry_count": self.entry_count,
                "exit_count": self.exit_count,
                "entry_direction": self.entry_direction,
                "exit_direction": self.exit_direction,
                "current_inside": max(0, self.entry_count - self.exit_count),
                "detection_mode": self.detection_mode,
                "line_orientation": self.line_orientation,
                "line_position": self.line_position
            }

    def set_config(self, mode=None, orientation=None, position=None):
        with self.lock:
            if mode is not None and mode in ["motion", "hog", "hybrid"]:
                self.detection_mode = mode
            if orientation is not None and orientation in ["vertical", "horizontal"]:
                self.line_orientation = orientation
            if position is not None:
                self.line_position = max(0.1, min(0.9, float(position)))

    def reset(self):
        with self.lock:
            self.entry_count = 0
            self.exit_count = 0
            self.entry_direction = None
            self.exit_direction = None
            self.objects = {}
            self.next_object_id = 0

    def release(self):
        self.running = False
        self.cap.release()
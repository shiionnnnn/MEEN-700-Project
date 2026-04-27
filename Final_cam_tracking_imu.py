
import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import time
import math
from collections import deque

# ============================================================
# USER CONFIG
# ============================================================
NN_W = 640
NN_H = 352

# Conservative defaults for USB2. The script probes actual USB speed
# and uses lower FPS automatically if the link negotiates at USB2/HIGH.
USB3_FPS = 12
USB2_FPS = 8

USB3_IMU_RATE_HZ = 100
USB2_IMU_RATE_HZ = 50

MIN_DEPTH_MM = 200
MAX_DEPTH_MM = 5000
CENTER_PATCH = 7

TARGET_LABEL = None  # Set to a COCO class id if you want to lock to one class
#banana 46
#cow 19 no workie

# Blob candidates (first existing path wins)
BLOB_CANDIDATES = [
    Path("yolov8n_coco_640x352.blob"),
    Path("yolov8n.blob"),
]

# HSV mask tuning
H_TOL = 12
S_TOL = 60
V_TOL = 60

MORPH_OPEN_K = 3
MORPH_CLOSE_K = 5
USE_LIVE_CENTER_REFERENCE = True

# Auto-capture settings
AUTO_CAPTURE_DURATION = 2.0
hsv_samples = deque()
last_capture_time = time.time()
auto_capture_active = False
selected_hsv_points = []

# UI
mask_overlay_enabled = True
show_mask_window = True

# Connection / stability
REQUIRE_USB3 = True
CONNECT_RETRIES = 2
RETRY_DELAY_SEC = 3.0

# Diagnostic mode: lighter pipeline to isolate USB / transport issues.
# When True, disables stereo depth, IMU, and IR projector.
SAFE_MODE = False

# Correct COCO class names (80)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

mouse_state = {"clicked": False, "pt": (0, 0)}


# ============================================================
# HELPERS
# ============================================================
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["clicked"] = True
        param["pt"] = (x, y)


def get_script_dir():
    if "__file__" in globals():
        return Path(__file__).parent.resolve()
    return Path.cwd().resolve()


def find_blob_path():
    base_dir = get_script_dir()
    for candidate in BLOB_CANDIDATES:
        p = candidate if candidate.is_absolute() else (base_dir / candidate)
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find a YOLO blob. Tried:\n" +
        "\n".join(str((base_dir / c).resolve()) for c in BLOB_CANDIDATES)
    )


def choose_best_detection(detections, frame_w, frame_h):
    candidates = []
    for d in detections:
        if TARGET_LABEL is not None and d.label != TARGET_LABEL:
            continue

        x1 = int(d.xmin * frame_w)
        y1 = int(d.ymin * frame_h)
        x2 = int(d.xmax * frame_w)
        y2 = int(d.ymax * frame_h)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        candidates.append((float(d.confidence), area, d))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def get_class_name(label_id):
    if 0 <= int(label_id) < len(COCO_CLASSES):
        return COCO_CLASSES[int(label_id)]
    return f"Class {label_id}"


def save_hsv_points(filename="hsv_points.txt"):
    with open(filename, "w") as f:
        for i, hsv in enumerate(selected_hsv_points):
            h, s, v = [int(x) for x in hsv]
            f.write(f"HSV {i+1}: H={h} S={s} V={v}\n")
    print(f"Saved {len(selected_hsv_points)} HSV points to {filename}")


def sample_patch_hsv(hsv_img, cx, cy, radius=2):
    h, w = hsv_img.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    patch = hsv_img[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return patch.reshape(-1, 3).mean(axis=0)


def make_single_hsv_mask(hsv_img, hsv_ref, h_tol=12, s_tol=60, v_tol=60):
    h_ref, s_ref, v_ref = [int(round(x)) for x in hsv_ref]

    s_low = max(0, s_ref - s_tol)
    s_high = min(255, s_ref + s_tol)
    v_low = max(0, v_ref - v_tol)
    v_high = min(255, v_ref + v_tol)

    h_low = h_ref - h_tol
    h_high = h_ref + h_tol

    if h_low < 0:
        mask1 = cv2.inRange(
            hsv_img,
            np.array([0, s_low, v_low], dtype=np.uint8),
            np.array([h_high, s_high, v_high], dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv_img,
            np.array([180 + h_low, s_low, v_low], dtype=np.uint8),
            np.array([179, s_high, v_high], dtype=np.uint8),
        )
        return cv2.bitwise_or(mask1, mask2)

    if h_high > 179:
        mask1 = cv2.inRange(
            hsv_img,
            np.array([h_low, s_low, v_low], dtype=np.uint8),
            np.array([179, s_high, v_high], dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv_img,
            np.array([0, s_low, v_low], dtype=np.uint8),
            np.array([h_high - 180, s_high, v_high], dtype=np.uint8),
        )
        return cv2.bitwise_or(mask1, mask2)

    return cv2.inRange(
        hsv_img,
        np.array([h_low, s_low, v_low], dtype=np.uint8),
        np.array([h_high, s_high, v_high], dtype=np.uint8),
    )


def keep_component_near_center(mask, center_x, center_y):
    if mask is None or mask.size == 0:
        return mask

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape[:2]
    center_x = int(np.clip(center_x, 0, w - 1))
    center_y = int(np.clip(center_y, 0, h - 1))
    chosen_label = labels[center_y, center_x]

    if chosen_label == 0:
        best_label = 0
        best_dist = float("inf")
        min_area = max(20, int(0.005 * w * h))

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            cx, cy = centroids[lbl]
            dist = (cx - center_x) ** 2 + (cy - center_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_label = lbl

        chosen_label = best_label

    if chosen_label == 0:
        return np.zeros_like(mask)

    filtered = np.zeros_like(mask)
    filtered[labels == chosen_label] = 255
    return filtered


def build_object_mask(roi_hsv, hsv_refs, center_x, center_y):
    if roi_hsv is None or roi_hsv.size == 0 or not hsv_refs:
        return None

    combined_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)

    for ref in hsv_refs:
        single = make_single_hsv_mask(
            roi_hsv,
            ref,
            h_tol=H_TOL,
            s_tol=S_TOL,
            v_tol=V_TOL,
        )
        combined_mask = cv2.bitwise_or(combined_mask, single)

    if MORPH_OPEN_K > 1:
        k = np.ones((MORPH_OPEN_K, MORPH_OPEN_K), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, k)

    if MORPH_CLOSE_K > 1:
        k = np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, k)

    combined_mask = keep_component_near_center(combined_mask, center_x, center_y)
    return combined_mask


def extract_mask_geometry(roi_mask):
    if roi_mask is None or roi_mask.size == 0 or cv2.countNonZero(roi_mask) == 0:
        return None, None, None

    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return None, None, None

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6:
        return cnt, None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    rect = cv2.minAreaRect(cnt)
    return cnt, (cx, cy), rect


def make_mask_overlay(roi_bgr, roi_mask):
    if roi_mask is None or cv2.countNonZero(roi_mask) == 0:
        return roi_bgr.copy()

    overlay = roi_bgr.copy()
    overlay[roi_mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(roi_bgr, 0.70, overlay, 0.30, 0)

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 255), 1)
    return blended


def masked_object_view(roi_bgr, roi_mask):
    if roi_mask is None or cv2.countNonZero(roi_mask) == 0:
        return roi_bgr.copy()
    return cv2.bitwise_and(roi_bgr, roi_bgr, mask=roi_mask)


def median_valid_depth(values_mm):
    if values_mm is None:
        return None

    valid = values_mm[(values_mm >= MIN_DEPTH_MM) & (values_mm <= MAX_DEPTH_MM)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def depth_at_point(depth_frame_mm, u, v, patch=CENTER_PATCH):
    if depth_frame_mm is None:
        return None

    r = patch // 2
    h, w = depth_frame_mm.shape[:2]
    x1 = max(0, u - r)
    y1 = max(0, v - r)
    x2 = min(w, u + r + 1)
    y2 = min(h, v + r + 1)

    roi = depth_frame_mm[y1:y2, x1:x2]
    return median_valid_depth(roi)


def depth_from_mask_or_roi(depth_roi_mm, roi_mask=None):
    if depth_roi_mm is None:
        return None

    if roi_mask is not None and roi_mask.shape[:2] == depth_roi_mm.shape[:2] and cv2.countNonZero(roi_mask) > 0:
        values = depth_roi_mm[roi_mask > 0]
        z_mm = median_valid_depth(values)
        if z_mm is not None:
            return z_mm

    return median_valid_depth(depth_roi_mm)


def pixel_to_camera_xyz(u, v, z_mm, fx, fy, cx, cy):
    z_m = z_mm / 1000.0
    x_m = (u - cx) * z_m / fx
    y_m = (v - cy) * z_m / fy
    return x_m, y_m, z_m


def estimate_metric_size_from_rect(rect, z_m, fx, fy):
    (_, _), (w_px, h_px), _ = rect
    width_m = (float(w_px) * z_m) / fx
    height_m = (float(h_px) * z_m) / fy
    return width_m, height_m


def estimate_metric_size_from_box(box_w_px, box_h_px, z_m, fx, fy):
    width_m = (float(box_w_px) * z_m) / fx
    height_m = (float(box_h_px) * z_m) / fy
    return width_m, height_m


def wrap_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def quaternion_to_euler_deg(i, j, k, real):
    x = float(i)
    y = float(j)
    z = float(k)
    w = float(real)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


def read_latest_packets_nonblocking(queue):
    latest = None
    while True:
        item = queue.tryGet()
        if item is None:
            break
        latest = item
    return latest


def update_imu_state(q_imu, imu_state, use_rotation_vector):
    if q_imu is None:
        return imu_state

    while True:
        imu_data = q_imu.tryGet()
        if imu_data is None:
            break

        for packet in imu_data.packets:
            # raw accel / gyro
            try:
                accel = packet.acceleroMeter
                imu_state["accel"] = (float(accel.x), float(accel.y), float(accel.z))
                accel_ts = accel.getTimestampDevice()
            except Exception:
                accel_ts = None

            try:
                gyro = packet.gyroscope
                gx = float(gyro.x)
                gy = float(gyro.y)
                gz = float(gyro.z)
                imu_state["gyro"] = (gx, gy, gz)
                gyro_ts = gyro.getTimestampDevice()
            except Exception:
                gx = gy = gz = 0.0
                gyro_ts = None

            if use_rotation_vector:
                try:
                    rv = packet.rotationVector
                    roll, pitch, yaw = quaternion_to_euler_deg(rv.i, rv.j, rv.k, rv.real)
                    imu_state["roll"] = roll
                    imu_state["pitch"] = pitch
                    imu_state["yaw"] = yaw
                    imu_state["orientation_valid"] = True
                    imu_state["rv_accuracy_rad"] = float(rv.rotationVectorAccuracy)
                    imu_state["last_orientation_ts"] = rv.getTimestampDevice()
                    continue
                except Exception:
                    pass

            # Fallback orientation estimate from raw accel + gyro
            if accel_ts is not None:
                ax, ay, az = imu_state["accel"]
                acc_roll = math.degrees(math.atan2(ay, az if abs(az) > 1e-6 else 1e-6))
                acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az) + 1e-6))

                dt = 0.0
                ts = gyro_ts or accel_ts
                prev_ts = imu_state["last_orientation_ts"]
                if prev_ts is not None and ts is not None:
                    dt = max(0.0, (ts - prev_ts).total_seconds())

                alpha = 0.98
                if prev_ts is None or dt <= 0.0:
                    imu_state["roll"] = acc_roll
                    imu_state["pitch"] = acc_pitch
                else:
                    imu_state["roll"] = alpha * (imu_state["roll"] + math.degrees(gx * dt)) + (1.0 - alpha) * acc_roll
                    imu_state["pitch"] = alpha * (imu_state["pitch"] + math.degrees(gy * dt)) + (1.0 - alpha) * acc_pitch
                    imu_state["yaw"] = wrap_angle_deg(imu_state["yaw"] + math.degrees(gz * dt))

                imu_state["orientation_valid"] = True
                imu_state["last_orientation_ts"] = ts

    return imu_state


def get_display_orientation(imu_state, imu_zero):
    roll = imu_state["roll"]
    pitch = imu_state["pitch"]
    yaw = imu_state["yaw"]

    if imu_zero["valid"]:
        roll = wrap_angle_deg(roll - imu_zero["roll"])
        pitch = wrap_angle_deg(pitch - imu_zero["pitch"])
        yaw = wrap_angle_deg(yaw - imu_zero["yaw"])

    return roll, pitch, yaw


def create_pipeline(blob_path, cam_fps, imu_rate_hz, use_rotation_vector, enable_depth=True, enable_imu=True):
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setPreviewSize(NN_W, NN_H)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(cam_fps)
    cam_rgb.setPreviewKeepAspectRatio(False)

    nn = pipeline.create(dai.node.YoloDetectionNetwork)
    nn.setBlobPath(str(blob_path))
    nn.setConfidenceThreshold(0.5)
    nn.setNumClasses(80)
    nn.setCoordinateSize(4)
    nn.setIouThreshold(0.5)
    nn.input.setBlocking(False)
    try:
        nn.setNumInferenceThreads(2)
    except Exception:
        pass

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")

    cam_rgb.preview.link(nn.input)
    cam_rgb.preview.link(xout_rgb.input)
    nn.out.link(xout_nn.input)

    if enable_depth:
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setFps(cam_fps)
        mono_right.setFps(cam_fps)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setExtendedDisparity(False)
        stereo.initialConfig.setConfidenceThreshold(200)
        try:
            stereo.setOutputSize(NN_W, NN_H)
        except Exception:
            pass

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        stereo.depth.link(xout_depth.input)

    if enable_imu:
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, imu_rate_hz)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, imu_rate_hz)
        if use_rotation_vector:
            imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, imu_rate_hz)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        xout_imu = pipeline.create(dai.node.XLinkOut)
        xout_imu.setStreamName("imu")
        imu.out.link(xout_imu.input)

    return pipeline


# ============================================================
# MAIN
# ============================================================
def main():
    global auto_capture_active, last_capture_time, mask_overlay_enabled

    blob_path = find_blob_path()
    print(f"Using blob: {blob_path}")
    if SAFE_MODE:
        print("SAFE_MODE is ON: depth, IMU, and IR projector are disabled for diagnostics.")

    # Give the USB stack a moment to settle before opening the device.
    time.sleep(1.0)

    enable_depth = not SAFE_MODE
    enable_imu = not SAFE_MODE

    empty_mask_view = np.zeros((NN_H, NN_W, 3), dtype=np.uint8)

    imu_state = {
        "accel": (0.0, 0.0, 0.0),
        "gyro": (0.0, 0.0, 0.0),
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "orientation_valid": False,
        "rv_accuracy_rad": None,
        "last_orientation_ts": None,
    }
    imu_zero = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "valid": False}

    last_err = None

    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            cam_fps = USB3_FPS
            imu_rate_hz = USB3_IMU_RATE_HZ
            use_rotation_vector = True
            pipeline = create_pipeline(
                blob_path,
                cam_fps,
                imu_rate_hz,
                use_rotation_vector,
                enable_depth=enable_depth,
                enable_imu=enable_imu,
            )

            with dai.Device(pipeline) as device:
                try:
                    usb_speed_obj = device.getUsbSpeed()
                    usb_speed_str = str(usb_speed_obj).split(".")[-1]
                except Exception:
                    usb_speed_obj = None
                    usb_speed_str = "UNKNOWN"

                try:
                    imu_type = str(device.getConnectedIMU()) if enable_imu else "DISABLED"
                except Exception:
                    imu_type = "UNKNOWN"

                print(f"Connection attempt {attempt}/{CONNECT_RETRIES}: USB speed = {usb_speed_str}")

                if REQUIRE_USB3:
                    super_speeds = {getattr(dai.UsbSpeed, "SUPER", None), getattr(dai.UsbSpeed, "SUPER_PLUS", None)}
                    super_speeds.discard(None)
                    if usb_speed_obj not in super_speeds:
                        raise RuntimeError(
                            f"Device negotiated at {usb_speed_str}, not USB3. Refusing to run this pipeline."
                        )

                use_rotation_vector = enable_imu and imu_type.startswith("BNO08")

                if not SAFE_MODE:
                    try:
                        device.setIrLaserDotProjectorBrightness(200)
                    except Exception:
                        try:
                            device.setIrLaserDotProjectorIntensity(0.5)
                        except Exception:
                            pass

                q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
                q_det = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
                q_depth = device.getOutputQueue(name="depth", maxSize=1, blocking=False) if enable_depth else None
                q_imu = device.getOutputQueue(name="imu", maxSize=10, blocking=False) if enable_imu else None

                calib = device.readCalibration()
                intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, NN_W, NN_H)
                fx = float(intr[0][0])
                fy = float(intr[1][1])
                cx = float(intr[0][2])
                cy = float(intr[1][2])

                print("YOLO + HSV Object Mask" + (" + Depth" if enable_depth else "") + (" + IMU" if enable_imu else ""))
                print(f"USB speed: {usb_speed_str}")
                print(f"IMU type: {imu_type}")
                print(f"Using rotation vector: {'YES' if use_rotation_vector else 'NO'}")
                print(f"RGB FPS target: {cam_fps}")
                print("Controls:")
                print("  a = toggle auto-capture")
                print("  s = save HSV points")
                print("  c = clear HSV points")
                print("  m = toggle mask overlay")
                print("  k = calibrate IMU zero")
                print("  q or ESC = quit")

                cv2.namedWindow("YOLO Detection")
                cv2.setMouseCallback("YOLO Detection", on_mouse, mouse_state)
                if show_mask_window:
                    cv2.namedWindow("HSV Object Mask")

                last_detections = []
                last_depth_mm = None

                while True:
                    in_rgb = q_rgb.tryGet()
                    if in_rgb is None:
                        continue
                    in_det = read_latest_packets_nonblocking(q_det)
                    in_depth = read_latest_packets_nonblocking(q_depth) if q_depth is not None else None

                    if in_det is not None:
                        last_detections = in_det.detections
                    if in_depth is not None:
                        last_depth_mm = in_depth.getFrame().astype(np.float32)

                    imu_state = update_imu_state(q_imu, imu_state, use_rotation_vector)

                    frame = in_rgb.getCvFrame()
                    if frame is None:
                        continue

                    clean_frame = frame.copy()
                    full_hsv = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2HSV)

                    h, w = frame.shape[:2]
                    current_time = time.time()

                    best = choose_best_detection(last_detections, w, h)

                    mask_view = empty_mask_view.copy()
                    mask_coverage = 0.0
                    roi_mask = None
                    roi_cnt = None
                    roi_rect = None
                    cu = cv = None
                    z_mm = None
                    box_w_px = box_h_px = None

                    if best is not None:
                        x1 = max(0, min(w - 1, int(best.xmin * w)))
                        y1 = max(0, min(h - 1, int(best.ymin * h)))
                        x2 = max(x1 + 1, min(w, int(best.xmax * w)))
                        y2 = max(y1 + 1, min(h, int(best.ymax * h)))

                        roi_bgr = clean_frame[y1:y2, x1:x2]
                        roi_hsv = full_hsv[y1:y2, x1:x2]
                        depth_roi = None
                        if last_depth_mm is not None and last_depth_mm.shape[:2] == (h, w):
                            depth_roi = last_depth_mm[y1:y2, x1:x2]

                        box_w_px = max(1, x2 - x1)
                        box_h_px = max(1, y2 - y1)
                        u = x1 + box_w_px // 2
                        v = y1 + box_h_px // 2

                        center_sample = sample_patch_hsv(full_hsv, u, v, radius=2)

                        reference_hsvs = list(selected_hsv_points)
                        using_live_ref = False
                        if not reference_hsvs and USE_LIVE_CENTER_REFERENCE and center_sample is not None:
                            reference_hsvs = [center_sample]
                            using_live_ref = True

                        roi_center_x = min(box_w_px - 1, max(0, u - x1))
                        roi_center_y = min(box_h_px - 1, max(0, v - y1))

                        if reference_hsvs and roi_hsv.size > 0:
                            roi_mask = build_object_mask(roi_hsv, reference_hsvs, roi_center_x, roi_center_y)

                        if roi_mask is not None and roi_mask.size > 0:
                            nonzero = cv2.countNonZero(roi_mask)
                            mask_coverage = 100.0 * nonzero / roi_mask.size if roi_mask.size > 0 else 0.0
                            roi_cnt, centroid_local, roi_rect = extract_mask_geometry(roi_mask)
                        else:
                            centroid_local = None

                        if centroid_local is not None:
                            cu = x1 + int(centroid_local[0])
                            cv = y1 + int(centroid_local[1])
                        else:
                            cu = u
                            cv = v

                        z_mm = None
                        if depth_roi is not None:
                            z_mm = depth_from_mask_or_roi(depth_roi, roi_mask)
                        if z_mm is None and enable_depth:
                            z_mm = depth_at_point(last_depth_mm, cu, cv)
                        if z_mm is None and enable_depth:
                            z_mm = depth_at_point(last_depth_mm, u, v)

                        # Auto-capture HSV
                        if auto_capture_active:
                            capture_sample = None

                            if roi_mask is not None and cv2.countNonZero(roi_mask) > 0:
                                masked_pixels = roi_hsv[roi_mask > 0]
                                if masked_pixels.size > 0:
                                    capture_sample = masked_pixels.mean(axis=0)

                            if capture_sample is None and center_sample is not None:
                                capture_sample = center_sample

                            if capture_sample is not None:
                                hsv_samples.append((current_time, capture_sample))

                            while hsv_samples and current_time - hsv_samples[0][0] > AUTO_CAPTURE_DURATION:
                                hsv_samples.popleft()

                            if current_time - last_capture_time >= AUTO_CAPTURE_DURATION and hsv_samples:
                                avg_hsv = np.mean([s[1] for s in hsv_samples], axis=0).astype(np.uint8)
                                selected_hsv_points.append(avg_hsv)
                                hsv_samples.clear()
                                last_capture_time = current_time
                                print(f"Captured HSV #{len(selected_hsv_points)}: {avg_hsv.tolist()}")

                        if mask_overlay_enabled and roi_mask is not None:
                            frame[y1:y2, x1:x2] = make_mask_overlay(roi_bgr, roi_mask)

                        # Draw primary box / center
                        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
                        cv2.circle(frame, (cu, cv), 4, (0, 0, 255), -1)

                        # Draw mask contour + min area rect if available
                        if roi_cnt is not None:
                            cnt_global = roi_cnt + np.array([[[x1, y1]]], dtype=np.int32)
                            cv2.drawContours(frame, [cnt_global], -1, (255, 255, 0), 2)

                        if roi_rect is not None:
                            box = cv2.boxPoints(roi_rect)
                            box[:, 0] += x1
                            box[:, 1] += y1
                            box = np.int32(box)
                            cv2.drawContours(frame, [box], 0, (0, 165, 255), 2)

                        # Text block
                        class_name = get_class_name(best.label)
                        line1 = f"{class_name} conf={best.confidence:.2f}"
                        line2 = f"box={box_w_px}x{box_h_px}px center=({cu},{cv})"

                        if z_mm is not None:
                            X, Y, Z = pixel_to_camera_xyz(cu, cv, z_mm, fx, fy, cx, cy)
                            range_m = math.sqrt(X * X + Y * Y + Z * Z)

                            if roi_rect is not None:
                                size_w_m, size_h_m = estimate_metric_size_from_rect(roi_rect, Z, fx, fy)
                            else:
                                size_w_m, size_h_m = estimate_metric_size_from_box(box_w_px, box_h_px, Z, fx, fy)

                            line3 = f"Z={Z:.3f}m range={range_m:.3f}m XYZ=({X:.2f},{Y:.2f},{Z:.2f})"
                            line4 = f"size~={size_w_m:.3f}m x {size_h_m:.3f}m mask={mask_coverage:.1f}%"
                        else:
                            line3 = "Depth invalid" if enable_depth else "Depth disabled (SAFE_MODE)"
                            line4 = f"mask={mask_coverage:.1f}%"

                        if using_live_ref:
                            line4 += "  liveHSV"

                        y_text = 25
                        for txt in [line1, line2, line3, line4]:
                            cv2.putText(frame, txt, (10, y_text),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
                            y_text += 22

                        if show_mask_window and roi_bgr.size > 0:
                            if roi_mask is not None and cv2.countNonZero(roi_mask) > 0:
                                mask_view = masked_object_view(roi_bgr, roi_mask)
                            else:
                                mask_view = roi_bgr.copy()
                            mask_view = cv2.resize(mask_view, (NN_W, NN_H), interpolation=cv2.INTER_NEAREST)
                    else:
                        cv2.putText(frame, "No object detected", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        if show_mask_window:
                            cv2.putText(mask_view, "No detection", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # IMU overlay
                    disp_roll, disp_pitch, disp_yaw = get_display_orientation(imu_state, imu_zero)
                    ax, ay, az = imu_state["accel"]
                    gx, gy, gz = imu_state["gyro"]

                    imu_line1 = f"ori r/p/y=({disp_roll:+.1f}, {disp_pitch:+.1f}, {disp_yaw:+.1f}) deg"
                    imu_line2 = f"accel=({ax:.2f}, {ay:.2f}, {az:.2f}) m/s^2"
                    imu_line3 = f"gyro=({gx:.3f}, {gy:.3f}, {gz:.3f}) rad/s"

                    if imu_state["rv_accuracy_rad"] is not None:
                        imu_line1 += f"  acc={math.degrees(imu_state['rv_accuracy_rad']):.1f}deg"

                    if enable_imu:
                        cv2.putText(frame, imu_line1, (10, h - 58),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
                        cv2.putText(frame, imu_line2, (10, h - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
                        cv2.putText(frame, imu_line3, (10, h - 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)

                    # Bottom status
                    auto_status = "CAPTURING" if auto_capture_active else "Ready"
                    samples_count = len([s for s in hsv_samples if current_time - s[0] < AUTO_CAPTURE_DURATION])
                    status = (
                        f"USB={usb_speed_str}  FPS={cam_fps}  SAFE={SAFE_MODE}  "
                        f"{auto_status}  samples={samples_count}  savedHSV={len(selected_hsv_points)}"
                    )
                    cv2.putText(frame, status, (10, h - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

                    # Draw clickable IMU calibrate button
                    bx1, by1, bx2, by2 = w - 185, 10, w - 10, 42
                    btn_color = (60, 160, 60) if imu_state["orientation_valid"] and enable_imu else (80, 80, 80)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_color, -1)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
                    cv2.putText(frame, "Calibrate IMU [K]", (bx1 + 10, by1 + 21),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    if mouse_state["clicked"]:
                        mx, my = mouse_state["pt"]
                        if bx1 <= mx <= bx2 and by1 <= my <= by2 and imu_state["orientation_valid"] and enable_imu:
                            imu_zero["roll"] = imu_state["roll"]
                            imu_zero["pitch"] = imu_state["pitch"]
                            imu_zero["yaw"] = imu_state["yaw"]
                            imu_zero["valid"] = True
                            print(
                                "IMU zero calibrated from button "
                                f"(roll={imu_zero['roll']:.2f}, pitch={imu_zero['pitch']:.2f}, yaw={imu_zero['yaw']:.2f})"
                            )
                        mouse_state["clicked"] = False

                    cv2.imshow("YOLO Detection", frame)
                    if show_mask_window:
                        cv2.imshow("HSV Object Mask", mask_view)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        return
                    elif key == ord("a"):
                        auto_capture_active = not auto_capture_active
                        hsv_samples.clear()
                        last_capture_time = time.time()
                        print(f"Auto-capture: {'ON' if auto_capture_active else 'OFF'}")
                    elif key == ord("c"):
                        selected_hsv_points.clear()
                        hsv_samples.clear()
                        print("Cleared all HSV points")
                    elif key == ord("s"):
                        if selected_hsv_points:
                            save_hsv_points("hsv_points.txt")
                        else:
                            print("No HSV points to save")
                    elif key == ord("m"):
                        mask_overlay_enabled = not mask_overlay_enabled
                        print(f"Mask overlay: {'ON' if mask_overlay_enabled else 'OFF'}")
                    elif key == ord("k"):
                        if imu_state["orientation_valid"] and enable_imu:
                            imu_zero["roll"] = imu_state["roll"]
                            imu_zero["pitch"] = imu_state["pitch"]
                            imu_zero["yaw"] = imu_state["yaw"]
                            imu_zero["valid"] = True
                            print(
                                "IMU zero calibrated from keyboard "
                                f"(roll={imu_zero['roll']:.2f}, pitch={imu_zero['pitch']:.2f}, yaw={imu_zero['yaw']:.2f})"
                            )
                        elif enable_imu:
                            print("IMU orientation is not valid yet; calibration ignored.")

        except RuntimeError as e:
            last_err = e
            print(f"Attempt {attempt}/{CONNECT_RETRIES} failed: {e}")
            if attempt < CONNECT_RETRIES:
                print(f"Retrying in {RETRY_DELAY_SEC:.1f}s...")
                time.sleep(RETRY_DELAY_SEC)
            continue
        except Exception as e:
            last_err = e
            raise
        finally:
            cv2.destroyAllWindows()

    raise RuntimeError(f"Unable to start stable OAK session after {CONNECT_RETRIES} attempts: {last_err}")


if __name__ == "__main__":
    main()

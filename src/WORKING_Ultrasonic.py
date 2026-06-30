#!/usr/bin/env python3
import time
import numpy as np
import serial

CFG = {
    "serial_port": "COM7",
    "baud_rate": 9600,

    "default_dt": 0.17,          # fallback if first timestamp
    "min_valid_cm": 1.0,
    "max_valid_cm": 400.0,

    # Soft jump handling
    "max_step_cm": 8.0,          # normal accepted one-step change
    "jump_confirm_tol_cm": 4.0,  # second jump reading must be near first candidate
    "predict_accept_tol_cm": 10.0,  # allow if close to KF predicted position

    # Kalman filter parameters
    "q_pos": 0.20,
    "q_vel": 1.50,
    "r_meas": 0.80,
    "p0": 5.0,
    "max_velocity_cm_s": 200.0,
}


class KalmanCV:
    """2-state constant-velocity Kalman filter: [distance, velocity]."""

    def __init__(self, x0, q_pos, q_vel, r_meas, p0):
        self.x = np.array([[x0], [0.0]], dtype=float)
        self.P = np.eye(2) * p0
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_meas = r_meas

    def predict(self, dt):
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)
        Q = np.array([[self.q_pos, 0.0],
                      [0.0, self.q_vel]], dtype=float)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.array([[1.0, 0.0]], dtype=float)
        R = np.array([[self.r_meas]], dtype=float)
        z = np.array([[z]], dtype=float)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

        return float(self.x[0, 0]), float(self.x[1, 0])

    def predicted_position(self):
        return float(self.x[0, 0])


class SensorFilter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_valid = None
        self.kf = None
        self.jump_candidate = None

    def _accept(self, z, dt):
        self.last_valid = float(z)
        self.jump_candidate = None

        if self.kf is None:
            self.kf = KalmanCV(
                x0=z,
                q_pos=self.cfg["q_pos"],
                q_vel=self.cfg["q_vel"],
                r_meas=self.cfg["r_meas"],
                p0=self.cfg["p0"],
            )

        self.kf.predict(dt)
        pos, vel = self.kf.update(z)

        vel = float(np.clip(vel,
                            -self.cfg["max_velocity_cm_s"],
                            self.cfg["max_velocity_cm_s"]))
        self.kf.x[1, 0] = vel

        return pos, vel, "ok"

    def process(self, z, dt):
        if z is None:
            return None, None, "missing"

        if z < self.cfg["min_valid_cm"] or z > self.cfg["max_valid_cm"]:
            return None, None, "bounds"

        if self.last_valid is None:
            return self._accept(z, dt)

        step = abs(z - self.last_valid)

        # Normal step: accept immediately
        if step <= self.cfg["max_step_cm"]:
            return self._accept(z, dt)

        # Accept a big jump if it is still near the predicted motion
        if self.kf is not None:
            pred = self.kf.predicted_position()
            if abs(z - pred) <= self.cfg["predict_accept_tol_cm"]:
                return self._accept(z, dt)

        # Otherwise use 2-sample confirmation
        if self.jump_candidate is None:
            self.jump_candidate = z
            return None, None, "jump_candidate"

        if abs(z - self.jump_candidate) <= self.cfg["jump_confirm_tol_cm"]:
            return self._accept(z, dt)

        self.jump_candidate = z
        return None, None, "jump_rejected"


class ArduinoSerialInput:
    """
    Expects:
    arduino_ms,object_raw_cm,table_raw_cm
    """

    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()

    def read(self):
        while True:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                return None, None, None

            if line.startswith("arduino_ms"):
                continue

            parts = line.split(",")
            if len(parts) != 3:
                return None, None, None

            try:
                arduino_ms = int(parts[0])
                object_cm = float(parts[1])
                table_cm = float(parts[2])

                if object_cm < 0:
                    object_cm = None
                if table_cm < 0:
                    table_cm = None

                return arduino_ms, object_cm, table_cm
            except ValueError:
                return None, None, None


def fmt(x, width=7):
    return "None".rjust(width) if x is None else f"{x:{width}.2f}"


def main():
    source = ArduinoSerialInput(CFG["serial_port"], CFG["baud_rate"])
    obj_filter = SensorFilter(CFG)
    tbl_filter = SensorFilter(CFG)

    last_arduino_ms = None

    print(f"Reading from {CFG['serial_port']} @ {CFG['baud_rate']}")
    #print(" ard_ms | obj_raw tbl_raw | obj_flt tbl_flt | height | obj_v  tbl_v")
    print(" obj_raw tbl_raw | obj_flt tbl_flt | height ")
    print("-" * 78)

    while True:
        arduino_ms, obj_raw, tbl_raw = source.read()
        if arduino_ms is None:
            continue

        if last_arduino_ms is None:
            dt = CFG["default_dt"]
        else:
            dt = max(0.001, (arduino_ms - last_arduino_ms) / 1000.0)
        last_arduino_ms = arduino_ms

        obj_f, obj_v, _ = obj_filter.process(obj_raw, dt)
        tbl_f, tbl_v, _ = tbl_filter.process(tbl_raw, dt)

        height = None
        if obj_f is not None and tbl_f is not None:
            height = abs(tbl_f - obj_f)

        print(
            #f"{str(arduino_ms).rjust(7)} | "
            f"object raw:{fmt(obj_raw)} table raw:{fmt(tbl_raw)} | "
            f"object filtered:{fmt(obj_f)} table filtered:{fmt(tbl_f)} | "
            f"height of object:{fmt(height)} | "
            #f"{fmt(obj_v)} {fmt(tbl_v)}"
        )


if __name__ == "__main__":
    main()

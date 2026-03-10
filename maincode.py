import cv2
import math
import time
import sys

try:
    import board
    import busio
    import RPi.GPIO as GPIO
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo as adafruit_servo
    HW_AVAILABLE = True
except ImportError:
    print("[WARN] Hardware libraries not found — running in DRY-RUN mode.")
    HW_AVAILABLE = False

from ultralytics import YOLO

# ══════════════════════════════════════════════
# CONFIGURATION — UPDATE FOR YOUR ROBOT
# ══════════════════════════════════════════════
# Arm lengths in centimetres
L1 =  8.0    # base-plate height to shoulder pivot
L2 = 8.5    # shoulder to elbow
L3 = 8.5    # elbow to magnet tip

# Camera calibration
# Measure: place object at known X cm from centre → count pixels → factor = cm/pixel
CALIB_X   = 0.05    # cm per pixel in X  (horizontal)
CALIB_Y   = 0.05    # cm per pixel in Y  (vertical — can differ from X)

CAMERA_W  = 640
CAMERA_H  = 480
CAM_CX    = CAMERA_W // 2     # image optical centre X
CAM_CY    = CAMERA_H // 2     # image optical centre Y

# Physical offset: how far (cm) is the camera centre ahead of arm base
CAMERA_OFFSET_Y = 15.0        # +Y = forward from arm base

# Table surface height relative to arm shoulder plane (cm)
TABLE_Z_CM = 0.0              # 0 = same height as shoulder; adjust if table is below

# Hardware
MAGNET_PIN    = 17            # BCM GPIO for relay
SERVO_CHANNEL = {             # PCA9685 channel mapping
    "base":     0,
    "shoulder": 1,
    "elbow":    2,
}
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

# FK acceptance threshold (cm)
FK_THRESHOLD_CM = 2.0

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Target hover height above table (cm)
HOVER_Z_CM = 2.0

# ══════════════════════════════════════════════
# HARDWARE INITIALISATION
# ══════════════════════════════════════════════
base_servo     = None
shoulder_servo = None
elbow_servo    = None

if HW_AVAILABLE:
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50

        base_servo     = adafruit_servo.Servo(pca.channels[SERVO_CHANNEL["base"]],
                                               min_pulse=SERVO_MIN_PULSE,
                                               max_pulse=SERVO_MAX_PULSE)
        shoulder_servo = adafruit_servo.Servo(pca.channels[SERVO_CHANNEL["shoulder"]],
                                               min_pulse=SERVO_MIN_PULSE,
                                               max_pulse=SERVO_MAX_PULSE)
        elbow_servo    = adafruit_servo.Servo(pca.channels[SERVO_CHANNEL["elbow"]],
                                               min_pulse=SERVO_MIN_PULSE,
                                               max_pulse=SERVO_MAX_PULSE)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(MAGNET_PIN, GPIO.OUT, initial=GPIO.LOW)
        print("[HW] PCA9685 + GPIO initialised OK")

    except Exception as e:
        print(f"[ERROR] Hardware init failed: {e}")
        sys.exit(1)

# ══════════════════════════════════════════════
# MOTION CONTROL
# ══════════════════════════════════════════════

def _set_servo(servo_obj, angle_deg, name="servo"):
    """Clamp and set one servo angle."""
    angle_deg = max(0.0, min(180.0, angle_deg))
    if HW_AVAILABLE and servo_obj is not None:
        servo_obj.angle = angle_deg
   
def move_robot(base_deg, shoulder_deg, elbow_deg, steps=20, delay=0.05):
    """
    Smoothly interpolate all three servos to target angles.
    steps  : number of intermediate positions (higher = smoother)
    delay  : seconds between steps
    """
    base_deg     = max(0.0, min(180.0, base_deg))
    shoulder_deg = max(0.0, min(180.0, shoulder_deg))
    elbow_deg    = max(0.0, min(180.0, elbow_deg))

    # Read current angles (default to 90 if unknown)
    cur_b = getattr(base_servo,     "angle", 90.0) or 90.0
    cur_s = getattr(shoulder_servo, "angle", 90.0) or 90.0
    cur_e = getattr(elbow_servo,    "angle", 90.0) or 90.0

    print(f"  [Move] Base:{cur_b:.0f}→{base_deg:.0f}  "
          f"Shld:{cur_s:.0f}→{shoulder_deg:.0f}  "
          f"Elbw:{cur_e:.0f}→{elbow_deg:.0f}  (steps={steps})")

    for i in range(1, steps + 1):
        t = i / steps
        _set_servo(base_servo,     cur_b + t * (base_deg     - cur_b))
        _set_servo(shoulder_servo, cur_s + t * (shoulder_deg - cur_s))
        _set_servo(elbow_servo,    cur_e + t * (elbow_deg    - cur_e))
        time.sleep(delay)

def toggle_magnet(state: bool):
    """True = ON (pick up), False = OFF (drop)."""
    label = "ON" if state else "OFF"
    if HW_AVAILABLE:
        GPIO.output(MAGNET_PIN, GPIO.HIGH if state else GPIO.LOW)
    print(f"  [Magnet] {label}")
    time.sleep(0.4)

# ══════════════════════════════════════════════
# KINEMATICS
# ══════════════════════════════════════════════

def inverse_kinematics(x_cm, y_cm, z_cm):
  
    try:
        # ── Yaw (base) ──
        theta1 = math.degrees(math.atan2(y_cm, x_cm))

        # ── Planar reach ──
        r      = math.sqrt(x_cm**2 + y_cm**2)
        z_new  = z_cm - L1                       # height above shoulder

        d      = math.sqrt(r**2 + z_new**2)

        # ── Reachability check ──
        if d > (L2 + L3) - 1e-4:
            print(f"  [IK] Out of reach (d={d:.2f} cm, max={L2+L3:.2f} cm)")
            return None

        # ── Elbow angle — CORRECTED cosine rule ──
        cos_t3 = (d**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
        cos_t3 = max(-1.0, min(1.0, cos_t3))     # numerical clamp
        theta3 = math.degrees(math.acos(cos_t3))

        # ── Shoulder angle — CORRECTED sign ──
        alpha  = math.degrees(math.atan2(z_new, r))
        beta   = math.degrees(math.acos(
                    max(-1.0, min(1.0,
                        (L2**2 + d**2 - L3**2) / (2.0 * L2 * d)
                    ))))
        theta2 = alpha - beta                   
        servo_base     = 90.0 + theta1           # 90° = centre/forward
        servo_shoulder = 90.0 - theta2           # 90° = arm horizontal
        servo_elbow    = 180.0 - theta3          # 180° = arm straight

        return servo_base, servo_shoulder, servo_elbow

    except Exception as e:
        print(f"  [IK] Math error: {e}")
        return None

def forward_kinematics(base_deg, shoulder_deg, elbow_deg):
    """
    FK for verification — returns (x, y, z) in cm from arm base.
    Inverse of the servo-angle mapping above.
    """
    theta1 = base_deg - 90.0
    theta2 = 90.0 - shoulder_deg
    theta3 = 180.0 - elbow_deg

    t1 = math.radians(theta1)
    t2 = math.radians(theta2)
    t3 = math.radians(theta3)

    rxy = L2 * math.cos(t2) + L3 * math.cos(t2 + t3)
    rz  = L2 * math.sin(t2) + L3 * math.sin(t2 + t3)

    x = rxy * math.cos(t1)
    y = rxy * math.sin(t1)
    z = L1  + rz

    return x, y, z

# ══════════════════════════════════════════════
# PIXEL → WORLD  (calibrated, no VP matrix needed on real cam)
# ══════════════════════════════════════════════
def pixel_to_world_cm(u, v):
    """
    Convert image pixel (u, v) to real-world X, Y in cm.
    Z is fixed at table height (TABLE_Z_CM) — no depth camera needed.
    Update CALIB_X / CALIB_Y / CAMERA_OFFSET_Y from your calibration.
    """
    x_cm = (u - CAM_CX) * CALIB_X
    y_cm = (CAM_CY - v) * CALIB_Y + CAMERA_OFFSET_Y   # flip V; add offset
    z_cm = TABLE_Z_CM
    return x_cm, y_cm, z_cm

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    cap = None
    try:
        # ── Load YOLO model ──
        print("Loading YOLO model ...")
        model = YOLO("best.pt")
        print("Model loaded.\n")

        # ── Home robot ──
        print("[Init] Homing robot ...")
        move_robot(90, 90, 90, steps=30)
        toggle_magnet(False)

        # ── Open camera ──
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera (index 0)")

        print("System ready. Press 'q' to quit.\n")
        print("═"*60)
        print("  AUTONOMOUS SORTING  —  Full Vision Pipeline (Hardware)")
        print("═"*60 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed — retrying ...")
                time.sleep(0.1)
                continue
            results = model(frame, stream=True, verbose=False)

            detections = []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    cv2.rectangle(frame,
                                  (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    detections.append((cx, cy, conf, label))

            if detections:
                u, v, conf, label = detections[0]

                print(f"{'─'*55}")
                print(f"  [YOLO]      {label}  conf={conf:.2f}  centroid=({u},{v})")
                x_cm, y_cm, z_cm = pixel_to_world_cm(u, v)
                print(f"  [Unproject] pixel ({u},{v})")
                print(f"              → World  X={x_cm:.2f}  Y={y_cm:.2f}  Z={z_cm:.2f} cm")
                angles = inverse_kinematics(x_cm, y_cm, HOVER_Z_CM)
                print(f"  [IK]        target Z={HOVER_Z_CM} cm (hover)")

                if angles is None:
                    print("  [IK]        UNREACHABLE — skip\n")
                    cv2.imshow("Mag-Scrap Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                b_deg, s_deg, e_deg = angles
                print(f"              → Base={b_deg:.1f}°  "
                      f"Shld={s_deg:.1f}°  Elbw={e_deg:.1f}°")
                fk_x, fk_y, fk_z = forward_kinematics(b_deg, s_deg, e_deg)
                fk_err = math.sqrt((fk_x - x_cm)**2 +
                                   (fk_y - y_cm)**2 +
                                   (fk_z - HOVER_Z_CM)**2)
                print(f"  [FK]        predicted  "
                      f"X={fk_x:.2f}  Y={fk_y:.2f}  Z={fk_z:.2f} cm")
                print(f"              IK→FK error = {fk_err:.3f} cm  ", end="")

                if fk_err > FK_THRESHOLD_CM:
                    print("TOO LARGE — skip\n")
                    cv2.imshow("Mag-Scrap Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                print("✔")
                print("  [Sequence]  Starting pickup ...")

                # A. Approach (hover above target)
                print("  [Move]      Approach (hover) ...")
                move_robot(b_deg, s_deg, e_deg, steps=25)
                time.sleep(0.5)

                # B. Lower Z to table surface & activate magnet
                angles_low = inverse_kinematics(x_cm, y_cm, 0.5)
                if angles_low:
                    print("  [Move]      Descend to surface ...")
                    move_robot(*angles_low, steps=15)
                toggle_magnet(True)
                time.sleep(0.5)

                # C. Lift back up (return to hover height)
                print("  [Move]      Lift ...")
                move_robot(b_deg, s_deg, e_deg, steps=15)
                time.sleep(0.3)

                # D. Rotate to drop zone (base = 0°)
                print("  [Move]      To drop zone ...")
                move_robot(0, 90, 90, steps=30)
                time.sleep(0.5)

                # E. Drop
                toggle_magnet(False)
                time.sleep(0.5)

                # F. Return home
                print("  [Move]      Return home ...")
                move_robot(90, 90, 90, steps=30)
                print("  [Done]      Pickup complete ✔\n")

                time.sleep(1.5)   # brief pause before next scan

            # Show vision feed
            cv2.imshow("Mag-Scrap Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit requested.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("\n[Cleanup] Releasing resources ...")
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        toggle_magnet(False)
        if HW_AVAILABLE:
            try:
                GPIO.cleanup()
                pca.deinit()
                print("[Cleanup] GPIO + PCA9685 released.")
            except Exception:
                pass
        print("[Cleanup] Done.")

if __name__ == "__main__":

    main()

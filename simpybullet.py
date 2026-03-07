"""
SIMULATION — Autonomous Robotic Sorting
========================================
Pipeline (matches real hardware pipeline exactly):
  PyBullet Virtual Camera
    → RGB frame  (colour classification)
    → Depth buffer (z-buffer → linearised metres)
      → Segmentation mask → Pixel Centroid
        → Depth value at centroid
          → Pixel-to-World (inverse VP matrix unproject)
            → Analytic IK  (same maths as hardware)
              → FK Verification
                → Move → Grasp (constraint) → Bin → Home

No OpenCV used — PyBullet virtual camera only.
"""

import pybullet as p
import pybullet_data
import numpy as np
import math
import time

# ══════════════════════════════════════════════
# CONFIGURATION  (mirrors hardware L1/L2/L3)
# ══════════════════════════════════════════════
L1 = 0.05   # base-to-shoulder height  (5 cm → metres)
L2 = 0.25   # shoulder-to-elbow        (matches sim arm)
L3 = 0.25   # elbow-to-tip

TABLE_Z  = 0.625
ARM_BASE = [0.0, 0.0, TABLE_Z]

# Camera
WIDTH, HEIGHT = 640, 480
FOV   = 60.0
NEAR  = 0.1
FAR   = 3.0

CAMERA_EYE    = [0.60,  0.00, 1.20]
CAMERA_TARGET = [0.28,  0.00, TABLE_Z]
CAMERA_UP     = [0.0,   1.0,  0.0]

BIN_POS     = [0.05, 0.38, TABLE_Z + 0.25]
HOME_ANGLES = [0.0, 0.3, -0.6]

FK_ERROR_THRESHOLD = 0.05   # metres

# ══════════════════════════════════════════════
# CONNECT
# ══════════════════════════════════════════════
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(
    cameraDistance=1.2, cameraYaw=50,
    cameraPitch=-30, cameraTargetPosition=[0.28, 0, 0.65])

# ══════════════════════════════════════════════
# ARM URDF
# ══════════════════════════════════════════════
URDF = f"""<?xml version="1.0"?>
<robot name="arm">

  <link name="base">
    <visual><geometry><box size="0.06 0.06 0.05"/></geometry>
      <material name="dk"><color rgba="0.2 0.2 0.2 1"/></material></visual>
    <collision><geometry><box size="0.06 0.06 0.05"/></geometry></collision>
    <inertial><mass value="1.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.025"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="200" velocity="2"/>
    <dynamics damping="0.5"/>
  </joint>
  <link name="l1">
    <inertial><mass value="0.5"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="2"/>
    <dynamics damping="0.5"/>
  </joint>
  <link name="l2">
    <visual><origin xyz="0 0 {L2/2}"/>
      <geometry><cylinder length="{L2}" radius="0.018"/></geometry>
      <material name="bl"><color rgba="0.2 0.4 0.9 1"/></material></visual>
    <collision><origin xyz="0 0 {L2/2}"/>
      <geometry><cylinder length="{L2}" radius="0.018"/></geometry></collision>
    <inertial><origin xyz="0 0 {L2/2}"/><mass value="0.3"/>
      <inertia ixx="0.003" iyy="0.003" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 {L2}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="200" velocity="2"/>
    <dynamics damping="0.5"/>
  </joint>
  <link name="l3">
    <visual><origin xyz="0 0 {L3/2}"/>
      <geometry><cylinder length="{L3}" radius="0.015"/></geometry>
      <material name="cy"><color rgba="0.1 0.85 0.85 1"/></material></visual>
    <collision><origin xyz="0 0 {L3/2}"/>
      <geometry><cylinder length="{L3}" radius="0.015"/></geometry></collision>
    <inertial><origin xyz="0 0 {L3/2}"/><mass value="0.2"/>
      <inertia ixx="0.002" iyy="0.002" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="ee_fixed" type="fixed">
    <parent link="l3"/><child link="ee"/>
    <origin xyz="0 0 {L3}"/>
  </joint>
  <link name="ee">
    <visual><geometry><sphere radius="0.016"/></geometry>
      <material name="rd"><color rgba="1 0.1 0.1 1"/></material></visual>
    <inertial><mass value="0.05"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
</robot>
"""
with open("arm.urdf", "w") as f:
    f.write(URDF)

# ══════════════════════════════════════════════
# LOAD WORLD
# ══════════════════════════════════════════════
p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", [0.4, 0, 0], useFixedBase=True)

try:
    p.loadURDF("tray/traybox.urdf", BIN_POS, globalScaling=0.5)
except Exception:
    bc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.04])
    bv = p.createVisualShape(p.GEOM_BOX,    halfExtents=[0.08, 0.08, 0.04],
                             rgbaColor=[0.55, 0.28, 0.08, 1])
    p.createMultiBody(0, bc, bv, BIN_POS)

arm = p.loadURDF("arm.urdf", ARM_BASE, useFixedBase=True)

# Auto-detect EE link index
print("═"*55)
print("  JOINT MAP")
print("═"*55)
EE_LINK = -1
for i in range(p.getNumJoints(arm)):
    info  = p.getJointInfo(arm, i)
    lname = info[12].decode()
    print(f"  joint {i}: {info[1].decode():<12} → link: {lname}")
    if lname == "ee":
        EE_LINK = i
print(f"  EE_LINK = {EE_LINK}")
print("═"*55 + "\n")

# ══════════════════════════════════════════════
# SPAWN OBJECTS  (scrap=grey cubes, non-scrap=gold spheres)
# ══════════════════════════════════════════════
scrap_ids   = []
all_objects = []

for i in range(8):
    pos = [
        np.random.uniform(0.18, 0.34),
        np.random.uniform(-0.13, 0.13),
        TABLE_Z + 0.06
    ]
    if i < 4:                               # SCRAP — grey cube
        try:
            obj = p.loadURDF("cube_small.urdf", pos)
        except Exception:
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
            v = p.createVisualShape(p.GEOM_BOX,    halfExtents=[0.02]*3)
            obj = p.createMultiBody(0.05, c, v, pos)
        p.changeVisualShape(obj, -1, rgbaColor=[0.35, 0.35, 0.35, 1])
        scrap_ids.append(obj)
    else:                                   # NON-SCRAP — gold sphere
        try:
            obj = p.loadURDF("sphere2.urdf", pos, globalScaling=0.05)
        except Exception:
            c = p.createCollisionShape(p.GEOM_SPHERE, radius=0.025)
            v = p.createVisualShape(p.GEOM_SPHERE,    radius=0.025)
            obj = p.createMultiBody(0.05, c, v, pos)
        p.changeVisualShape(obj, -1, rgbaColor=[1.0, 0.8, 0.0, 1])
    all_objects.append(obj)

print("Settling physics ...")
for _ in range(300):
    p.stepSimulation()
    time.sleep(1/240)
print("Done.\n")

# ══════════════════════════════════════════════
# CAMERA MATRICES
# ══════════════════════════════════════════════
VIEW_MAT = p.computeViewMatrix(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP)
PROJ_MAT = p.computeProjectionMatrixFOV(FOV, WIDTH/HEIGHT, NEAR, FAR)

# Pre-compute inverse VP for unprojection (standard CV approach)
V      = np.array(VIEW_MAT, dtype=np.float64).reshape(4, 4).T
P      = np.array(PROJ_MAT, dtype=np.float64).reshape(4, 4).T
VP_INV = np.linalg.inv(P @ V)

# ──────────────────────────────────────────────
# STEP A: Depth linearisation
#   PyBullet depth buffer stores non-linear z ∈ [0,1]
#   True depth (m) = FAR·NEAR / (FAR − (FAR−NEAR)·z_buf)
# ──────────────────────────────────────────────
def linearise_depth(z_buf):
    return (FAR * NEAR) / (FAR - (FAR - NEAR) * np.asarray(z_buf))

# ──────────────────────────────────────────────
# STEP B: Pixel → World (inverse VP matrix)
#   1. Convert pixel (u,v) → NDC [-1,1]
#   2. Reconstruct NDC z from linear depth
#   3. Unproject: world = VP⁻¹ · [x_ndc, y_ndc, z_ndc, 1]ᵀ
# ──────────────────────────────────────────────
def pixel_to_world(u, v, depth_m):
    x_ndc = (u + 0.5) / WIDTH  *  2.0 - 1.0
    y_ndc = 1.0 - (v + 0.5) / HEIGHT * 2.0     # flip Y (image→NDC)
    z_ndc = (FAR + NEAR - 2.0*FAR*NEAR / depth_m) / (FAR - NEAR)

    pt_ndc  = np.array([x_ndc, y_ndc, z_ndc, 1.0])
    pt_world = VP_INV @ pt_ndc
    pt_world /= pt_world[3]                      # perspective divide
    return pt_world[:3]

# ──────────────────────────────────────────────
# STEP C: Analytic IK  (identical maths to hardware)
#   θ1 = atan2(y, x)                 — yaw
#   r  = √(x²+y²),  z_new = z − L1  — planar reach
#   cos θ3 = (d²−L2²−L3²)/(2·L2·L3) — elbow (cosine rule) ← FIXED
#   α  = atan2(z_new, r)
#   β  = atan2(L3·sin θ3, L2+L3·cos θ3)
#   θ2 = α − β                       — shoulder ← FIXED
# ──────────────────────────────────────────────
def IK(x, y, z):
    dx = x - ARM_BASE[0]
    dy = y - ARM_BASE[1]
    dz = z - ARM_BASE[2] - L1      # height above shoulder

    t1 = math.atan2(dy, dx)

    r  = math.sqrt(dx**2 + dy**2)
    d  = math.sqrt(r**2  + dz**2)

    if d > (L2 + L3) - 1e-4:
        return None

    # ── Elbow angle (cosine rule — CORRECTED sign) ──
    cos_t3 = (d**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    cos_t3 = float(np.clip(cos_t3, -1.0, 1.0))
    t3 = math.acos(cos_t3)

    # ── Shoulder angle (CORRECTED: alpha − beta) ──
    alpha = math.atan2(dz, r)
    beta  = math.atan2(L3 * math.sin(t3), L2 + L3 * math.cos(t3))
    t2    = alpha - beta

    return [t1, t2, t3]

# ──────────────────────────────────────────────
# STEP D: FK verification
# ──────────────────────────────────────────────
def FK(angles):
    t1, t2, t3 = angles
    rxy = L2 * math.cos(t2) + L3 * math.cos(t2 + t3)
    rz  = L2 * math.sin(t2) + L3 * math.sin(t2 + t3)
    return [
        ARM_BASE[0] + rxy * math.cos(t1),
        ARM_BASE[1] + rxy * math.sin(t1),
        ARM_BASE[2] + L1  + rz
    ]

# ──────────────────────────────────────────────
# MOVE (joint position control)
# ──────────────────────────────────────────────
def move(angles, steps=300):
    if angles is None:
        return False
    for i, ang in enumerate(angles):
        p.setJointMotorControl2(arm, i, p.POSITION_CONTROL,
                                targetPosition=ang,
                                force=200, maxVelocity=1.5)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1/240)
    return True

# ══════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════
move(HOME_ANGLES, steps=200)

# ══════════════════════════════════════════════
# AUTONOMOUS SORT LOOP
# ══════════════════════════════════════════════
print("═"*60)
print("  AUTONOMOUS SORTING  —  Full Vision Pipeline  (Simulation)")
print("═"*60 + "\n")

remaining = scrap_ids[:]

while len(remaining) > 0:

    # ┌─────────────────────────────────────────────────┐
    # │ STEP 1 — Capture virtual camera frame           │
    # └─────────────────────────────────────────────────┘
    img_out = p.getCameraImage(WIDTH, HEIGHT, VIEW_MAT, PROJ_MAT,
                               renderer=p.ER_TINY_RENDERER)

    rgb_raw   = np.array(img_out[2], dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    depth_buf = np.array(img_out[3], dtype=np.float32).reshape(HEIGHT, WIDTH)
    seg_raw   = np.array(img_out[4], dtype=np.int32).reshape(HEIGHT, WIDTH)

    # ┌─────────────────────────────────────────────────┐
    # │ STEP 2 — Linearise depth buffer → metres        │
    # └─────────────────────────────────────────────────┘
    depth_m = linearise_depth(depth_buf)
    print(f"[Camera]  RGB {WIDTH}×{HEIGHT}  |  "
          f"Depth {depth_m.min():.3f}–{depth_m.max():.3f} m")

    picked_this_cycle = False

    for obj in remaining[:]:

        # ┌─────────────────────────────────────────────┐
        # │ STEP 3 — Segmentation mask                  │
        # └─────────────────────────────────────────────┘
        mask = (seg_raw == obj)
        if not np.any(mask):
            continue

        # ┌─────────────────────────────────────────────┐
        # │ STEP 4 — Pixel centroid                     │
        # └─────────────────────────────────────────────┘
        rows, cols = np.where(mask)
        cy = int(np.mean(rows))
        cx = int(np.mean(cols))

        # ┌─────────────────────────────────────────────┐
        # │ STEP 5 — RGB colour at centroid             │
        # │          Grey  → scrap                      │
        # │          Other → non-scrap, skip            │
        # └─────────────────────────────────────────────┘
        r_v = int(rgb_raw[cy, cx, 0])
        g_v = int(rgb_raw[cy, cx, 1])
        b_v = int(rgb_raw[cy, cx, 2])

        is_gray = (abs(r_v - g_v) < 25 and
                   abs(r_v - b_v) < 25 and
                   abs(g_v - b_v) < 25)

        print(f"\n{'─'*55}")
        print(f"  Object ID  : {obj}")
        print(f"  [RGB]       centroid pixel ({cx},{cy})  "
              f"→  R={r_v}  G={g_v}  B={b_v}")

        if not is_gray:
            print(f"  [Vision]    NON-SCRAP (coloured)  ✘  — ignored")
            continue

        print(f"  [Vision]    SCRAP (grey)  ✔")

        # ┌─────────────────────────────────────────────┐
        # │ STEP 6 — Depth value at centroid pixel      │
        # └─────────────────────────────────────────────┘
        d_buf = float(depth_buf[cy, cx])
        d_lin = float(depth_m[cy, cx])
        print(f"  [Depth]     pixel ({cx},{cy})  →  "
              f"z_buffer = {d_buf:.4f}  →  depth = {d_lin:.4f} m")

        # ┌─────────────────────────────────────────────┐
        # │ STEP 7 — Pixel → World (inverse VP matrix)  │
        # └─────────────────────────────────────────────┘
        w = pixel_to_world(cx, cy, d_lin)
        wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
        print(f"  [Unproject] pixel ({cx},{cy}) + {d_lin:.4f} m")
        print(f"              → World  X={wx:.4f}  Y={wy:.4f}  Z={wz:.4f} m")

        # Clamp Z to table surface (noise guard)
        wz = max(wz, TABLE_Z)

        # ┌─────────────────────────────────────────────┐
        # │ STEP 8 — Analytic IK                        │
        # └─────────────────────────────────────────────┘
        approach_z  = wz + 0.14
        ang_appr    = IK(wx, wy, approach_z)
        print(f"  [IK]        target  X={wx:.4f}  Y={wy:.4f}  Z={approach_z:.4f}")
        if ang_appr is None:
            print(f"  [IK]        UNREACHABLE — skip")
            continue
        print(f"              → j1={math.degrees(ang_appr[0]):+.1f}°  "
              f"j2={math.degrees(ang_appr[1]):+.1f}°  "
              f"j3={math.degrees(ang_appr[2]):+.1f}°")

        # ┌─────────────────────────────────────────────┐
        # │ STEP 9 — FK Verification                    │
        # └─────────────────────────────────────────────┘
        fk_pos = FK(ang_appr)
        ik_err = float(np.linalg.norm(
            np.array(fk_pos) - np.array([wx, wy, approach_z])))
        print(f"  [FK]        predicted EE  "
              f"X={fk_pos[0]:.4f}  Y={fk_pos[1]:.4f}  Z={fk_pos[2]:.4f}")
        print(f"              IK→FK error = {ik_err:.5f} m  ", end="")
        if ik_err > FK_ERROR_THRESHOLD:
            print("TOO LARGE — skip")
            continue
        print("✔")

        # ┌─────────────────────────────────────────────┐
        # │ STEP 10 — Move to approach                  │
        # └─────────────────────────────────────────────┘
        print(f"  [Move]      Approach ...")
        move(ang_appr)

        # ┌─────────────────────────────────────────────┐
        # │ STEP 11 — Descend & grasp                   │
        # └─────────────────────────────────────────────┘
        grasp_z  = wz + 0.04
        ang_grsp = IK(wx, wy, grasp_z)
        if ang_grsp is None:
            print("  [IK]        Grasp descent unreachable — skip")
            move(HOME_ANGLES); continue

        print(f"  [Move]      Descend to grasp ...")
        move(ang_grsp, steps=180)

        # Attach constraint — positional args only (no kwargs)
        cid = p.createConstraint(
            arm, EE_LINK,
            obj, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0.02],
            [0, 0, 0]
        )
        p.changeConstraint(cid, maxForce=500)
        print(f"  [Grasp]     Constraint attached  ✔")

        # ┌─────────────────────────────────────────────┐
        # │ STEP 12 — Lift                              │
        # └─────────────────────────────────────────────┘
        ang_lift = IK(wx, wy, wz + 0.30)
        if not move(ang_lift, steps=250):
            p.removeConstraint(cid); move(HOME_ANGLES); continue
        print(f"  [Move]      Lifted  ✔")

        # ┌─────────────────────────────────────────────┐
        # │ STEP 13 — Move to bin & release             │
        # └─────────────────────────────────────────────┘
        print(f"  [Move]      To bin {BIN_POS} ...")
        ang_bin = IK(*BIN_POS)
        if not move(ang_bin, steps=400):
            p.removeConstraint(cid); move(HOME_ANGLES); continue

        p.removeConstraint(cid)
        for _ in range(80):
            p.stepSimulation(); time.sleep(1/240)
        print(f"  [Bin]       Released  ✔")

        remaining.remove(obj)
        picked_this_cycle = True

        # ┌─────────────────────────────────────────────┐
        # │ STEP 14 — Return home                       │
        # └─────────────────────────────────────────────┘
        print(f"  [Move]      Return home ...")
        move(HOME_ANGLES, steps=250)
        break

    if not picked_this_cycle:
        print("\nNo more reachable / visible scrap — exiting loop.")
        break

print("\n" + "═"*60)
print("  ALL SCRAP REMOVED  ✔")
print("═"*60)
time.sleep(4)
p.disconnect()
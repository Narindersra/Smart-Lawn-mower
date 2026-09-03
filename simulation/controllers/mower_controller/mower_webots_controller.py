import sys
from pathlib import Path
import yaml
from controller import Robot
from raspberry_pi.src.localization import localization_manager, odometry
from raspberry_pi.src.safety import safety_manager
from safety_manager import SafetyManager
from localization.localization_manager import LocalizationManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from webots_camera import WebotsCameraAdapter
from ai.inference.inference import InferenceEngine


def run_simulation():
    robot = Robot()
    # Load safety configuration
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "config" / "development" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    ai_stop_classes = set(
        config["safety"]["ai_stop_classes"]
    )

    obstacle_distance_threshold = config["safety"]["obstacle_distance_threshold"]
    ai_confidence_threshold = config["safety"]["ai_confidence_threshold"]

    safety_manager = SafetyManager(
        ai_stop_classes,
        obstacle_distance_threshold,
    )

    timestep = int(robot.getBasicTimeStep())

    # --------------------------------------------------
    # IMU
    # --------------------------------------------------
    imu = robot.getDevice("imu")
    

    # --------------------------------------------------
    # GPS
    # --------------------------------------------------
    gps = robot.getDevice("gps")
    

    # AI Inference
    inference_engine = InferenceEngine(
        confidence_threshold=ai_confidence_threshold
    )

    # AI Safety State
    ai_safety_stop = False

    # --------------------------------------------------
    # Motors
    # --------------------------------------------------
    left_motor = robot.getDevice("left_wheel_motor")
    right_motor = robot.getDevice("right_wheel_motor")

    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # --------------------------------------------------
    # Front Distance Sensor
    # --------------------------------------------------
    ds_front = robot.getDevice("ds_front")
    ds_front.enable(timestep)

    # --------------------------------------------------
    # Camera
    # --------------------------------------------------
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    camera_adapter = WebotsCameraAdapter(camera)
    camera_adapter.initialize(timestep)

    # --------------------------------------------------
    # Wheel Encoders
    # --------------------------------------------------
    left_encoder = robot.getDevice("left_wheel_encoder")
    right_encoder = robot.getDevice("right_wheel_encoder")

    left_encoder.enable(timestep)
    right_encoder.enable(timestep)


    gps_origin = config["localization"]["gps_origin"]

    localization_manager = LocalizationManager(
        wheel_radius=0.08,
        wheel_track=0.44,
        gps_origin_x=gps_origin["x"],
        gps_origin_y=gps_origin["y"],
    )

    localization_manager.initialize(
        gps=gps,
        imu=imu,
        timestep=timestep,
    )

    # --------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------
    while robot.step(timestep) != -1:

        # Read front distance sensor
        distance = ds_front.getValue()
        obstacle_detected = safety_manager.should_stop_for_obstacle(distance)

        # Read wheel encoder positions
        left_encoder_position = left_encoder.getValue()
        right_encoder_position = right_encoder.getValue()

        odometry_data = odometry.update(
            left_encoder_position,
            right_encoder_position,
        )


       
     

        pose = localization_manager.update(
            left_encoder_position=left_encoder_position,
            right_encoder_position=right_encoder_position,
        )

        localization_ready = localization_manager.is_ready()

        # Camera Frame
        frame = camera_adapter.get_frame()

        # AI Object Detection
        detections = []

        if frame is not None:
            detections = inference_engine.run(frame)

        # AI Safety Check
        ai_safety_stop = safety_manager.should_emergency_stop(detections)

        # --------------------------------------------------
        # Basic Movement / Obstacle Avoidance
        # --------------------------------------------------
        if not localization_ready:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
        elif ai_safety_stop:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
        elif obstacle_detected:
            left_motor.setVelocity(-2.0)
            right_motor.setVelocity(2.0)
        else:
            left_motor.setVelocity(3.0)
            right_motor.setVelocity(3.0)


if __name__ == "__main__":
    run_simulation()
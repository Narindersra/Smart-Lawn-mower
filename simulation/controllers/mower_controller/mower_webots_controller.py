import sys
from pathlib import Path

import yaml
from controller import Robot


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from localization.localization_manager import LocalizationManager

from navigation.planner import NavigationPlanner
from navigation.navigation_types import NavigationState, Waypoint
from navigation.differential_drive import DifferentialDriveController
from navigation.obstacle_avoidance import ObstacleAvoidance
from navigation.obstacle_types import Obstacle, ObstacleInformation
from navigation.geofence import Geofence

from safety.safety_manager import SafetyManager

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

    geofence_config = config["geofence"]

    geofence = Geofence(
        min_x=geofence_config["min_x"],
        max_x=geofence_config["max_x"],
        min_y=geofence_config["min_y"],
        max_y=geofence_config["max_y"],
    )

    navigation_planner = NavigationPlanner(
        waypoint_tolerance=0.15,
        geofence=geofence,
    )

    drive_controller = DifferentialDriveController(
        wheel_radius=0.08,
        wheel_track=0.44,
    )

    obstacle_avoidance = ObstacleAvoidance()

    navigation_planner.set_waypoint(
        Waypoint(
            x=5.0,
            y=5.0,
        )
    )

    # --------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------
    while robot.step(timestep) != -1:

        # Read front distance sensor
        distance = ds_front.getValue() / 1000.0
        obstacle_detected = safety_manager.should_stop_for_obstacle(distance)
        obstacle_information = ObstacleInformation(
            obstacles=()
        )
        if obstacle_detected:
            navigation_planner.transition_to(
                NavigationState.AVOIDING
            )

        if (
            navigation_planner.state == NavigationState.AVOIDING
            and obstacle_avoidance.is_clear(obstacle_information)
        ):
            navigation_planner.transition_to(
                NavigationState.REPLANNING
            )
        

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

        if navigation_planner.state == NavigationState.REPLANNING:
            navigation_planner.replan(pose)

        localization_ready = localization_manager.is_ready()

        navigation_state, distance_to_waypoint, heading_error, motion_command = (
            navigation_planner.update(pose)
        )

        left_wheel_velocity, right_wheel_velocity = (
            drive_controller.calculate_wheel_velocities(
                linear_velocity=motion_command.linear_velocity,
                angular_velocity=motion_command.angular_velocity,
            )
        )

        avoidance_command = obstacle_avoidance.calculate(
            obstacle_information
        )

        avoidance_left_velocity, avoidance_right_velocity = (
            drive_controller.calculate_wheel_velocities(
                linear_velocity=avoidance_command.linear_velocity,
                angular_velocity=avoidance_command.angular_velocity,
            )
        )

        

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
            left_motor.setVelocity(avoidance_left_velocity)
            right_motor.setVelocity(avoidance_right_velocity)
        
        elif navigation_state == NavigationState.GOAL_REACHED:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
        
        elif navigation_state == NavigationState.EMERGENCY_STOP:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
        
        else:
            left_motor.setVelocity(left_wheel_velocity)
            right_motor.setVelocity(right_wheel_velocity)


if __name__ == "__main__":
    run_simulation()
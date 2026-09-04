import math
import unittest

from localization.gps import GPSReader
from localization.imu import IMUReader
from localization.odometry import DifferentialDriveOdometry
from localization.position_estimator import PositionEstimator, RobotPose
from raspberry_pi.src.localization.localization_manager import LocalizationManager


class FakeGPS:
    def __init__(self, values=(0.0, 0.0, 0.0)):
        self.values = values
        self.enabled_timestep = None

    def enable(self, timestep):
        self.enabled_timestep = timestep

    def getValues(self):
        return self.values


class FakeIMU:
    def __init__(self, yaw=0.0):
        self.yaw = yaw
        self.enabled_timestep = None

    def enable(self, timestep):
        self.enabled_timestep = timestep

    def getRollPitchYaw(self):
        return 0.0, 0.0, self.yaw


class TestGPSReader(unittest.TestCase):

    def test_initialization(self):
        gps_device = FakeGPS()
        gps = GPSReader(
            gps_device,
            origin_x=10.0,
            origin_y=20.0,
        )

        self.assertIs(gps.gps, gps_device)
        self.assertEqual(gps.origin_x, 10.0)
        self.assertEqual(gps.origin_y, 20.0)

    def test_initialize(self):
        gps_device = FakeGPS()
        gps = GPSReader(gps_device)

        gps.initialize(32)

        self.assertEqual(
            gps_device.enabled_timestep,
            32,
        )

    def test_get_position_with_local_origin(self):
        gps_device = FakeGPS(
            values=(12.0, 3.0, 23.0)
        )

        gps = GPSReader(
            gps_device,
            origin_x=10.0,
            origin_y=20.0,
        )

        position = gps.get_position()

        self.assertAlmostEqual(position[0], 2.0)
        self.assertAlmostEqual(position[1], 3.0)
        self.assertAlmostEqual(position[2], 3.0)


class TestIMUReader(unittest.TestCase):

    def test_initialization(self):
        imu_device = FakeIMU(
            yaw=0.5
        )

        imu = IMUReader(
            imu_device
        )

        self.assertIs(
            imu.imu,
            imu_device,
        )

    def test_initialize(self):
        imu_device = FakeIMU()
        imu = IMUReader(imu_device)

        imu.initialize(32)

        self.assertEqual(
            imu_device.enabled_timestep,
            32,
        )

    def test_get_heading(self):
        imu_device = FakeIMU(
            yaw=math.pi / 4
        )

        imu = IMUReader(
            imu_device
        )

        heading = imu.get_heading()

        self.assertAlmostEqual(
            heading,
            math.pi / 4,
        )


class TestOdometry(unittest.TestCase):

    def setUp(self):
        self.odometry = DifferentialDriveOdometry(
            wheel_radius=0.08,
            wheel_track=0.44,
        )

    def test_initialization(self):
        self.assertAlmostEqual(
            self.odometry.wheel_radius,
            0.08,
        )

        self.assertAlmostEqual(
            self.odometry.wheel_track,
            0.44,
        )

        self.assertIsNone(
            self.odometry.previous_left_position
        )

        self.assertIsNone(
            self.odometry.previous_right_position
        )

    def test_first_update_returns_zero_motion(self):
        result = self.odometry.update(
            left_encoder_position=0.0,
            right_encoder_position=0.0,
        )

        self.assertEqual(
            result["left_distance"],
            0.0,
        )

        self.assertEqual(
            result["right_distance"],
            0.0,
        )

        self.assertEqual(
            result["distance"],
            0.0,
        )

        self.assertEqual(
            result["heading_change"],
            0.0,
        )

    def test_forward_motion(self):
        self.odometry.update(
            left_encoder_position=0.0,
            right_encoder_position=0.0,
        )
    
        result = self.odometry.update(
            left_encoder_position=1.0,
            right_encoder_position=1.0,
        )
    
        self.assertGreater(
            result["left_distance"],
            0.0,
        )
    
        self.assertGreater(
            result["right_distance"],
            0.0,
        )
    
        self.assertGreater(
            result["distance"],
            0.0,
        )
    
        self.assertAlmostEqual(
            result["heading_change"],
            0.0,
        )

    def test_rotation(self):
        self.odometry.update(
            left_encoder_position=0.0,
            right_encoder_position=0.0,
        )

        result = self.odometry.update(
            left_encoder_position=-1.0,
            right_encoder_position=1.0,
        )

        self.assertNotEqual(
            result["heading_change"],
            0.0,
        )

        self.assertAlmostEqual(
            result["distance"],
            0.0,
        )

    def test_encoder_positions_are_stored(self):
        self.odometry.update(
            left_encoder_position=1.5,
            right_encoder_position=2.0,
        )

        self.assertAlmostEqual(
            self.odometry.previous_left_position,
            1.5,
        )

        self.assertAlmostEqual(
            self.odometry.previous_right_position,
            2.0,
        )


class TestRobotPose(unittest.TestCase):

    def test_robot_pose_values(self):
        pose = RobotPose(
            x=2.0,
            y=3.0,
            heading=1.0,
        )

        self.assertAlmostEqual(
            pose.x,
            2.0,
        )

        self.assertAlmostEqual(
            pose.y,
            3.0,
        )

        self.assertAlmostEqual(
            pose.heading,
            1.0,
        )


class TestPositionEstimator(unittest.TestCase):

    def test_initialization(self):
        estimator = PositionEstimator()

        pose = estimator.get_pose()

        self.assertIsInstance(
            pose,
            RobotPose,
        )

        self.assertAlmostEqual(
            pose.x,
            0.0,
        )

        self.assertAlmostEqual(
            pose.y,
            0.0,
        )

        self.assertAlmostEqual(
            pose.heading,
            0.0,
        )

    def test_custom_initial_pose(self):
        estimator = PositionEstimator(
            initial_x=2.0,
            initial_y=3.0,
            initial_heading=1.0,
        )

        pose = estimator.get_pose()

        self.assertAlmostEqual(
            pose.x,
            2.0,
        )

        self.assertAlmostEqual(
            pose.y,
            3.0,
        )

        self.assertAlmostEqual(
            pose.heading,
            1.0,
        )

    def test_gps_update(self):
        estimator = PositionEstimator()

        estimator.update_gps(
            1.0,
            2.0,
        )

        pose = estimator.get_pose()

        self.assertAlmostEqual(
            pose.x,
            1.0,
        )

        self.assertAlmostEqual(
            pose.y,
            2.0,
        )

    def test_imu_update(self):
        estimator = PositionEstimator()

        estimator.update_imu(
            0.5
        )

        pose = estimator.get_pose()

        self.assertAlmostEqual(
            pose.heading,
            0.5,
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )



class TestLocalizationManager(unittest.TestCase):

    def setUp(self):
        self.gps = FakeGPS(
            values=(2.0, 4.0, 0.0)
        )

        self.imu = FakeIMU(
            yaw=0.5,
        )

        self.manager = LocalizationManager(
            wheel_radius=0.08,
            wheel_track=0.44,
            gps_origin_x=0.0,
            gps_origin_y=0.0,
        )

    def test_manager_initial_state(self):
        self.assertFalse(
            self.manager.is_ready()
        )

    def test_manager_initialization(self):
        self.manager.initialize(
            gps=self.gps,
            imu=self.imu,
            timestep=32,
        )

        self.assertIsNotNone(
            self.manager.gps_reader
        )

        self.assertIsNotNone(
            self.manager.imu_reader
        )

        self.assertIsNotNone(
            self.manager.odometry
        )

        self.assertIsNotNone(
            self.manager.position_estimator
        )

    def test_manager_update(self):
        self.manager.initialize(
            gps=self.gps,
            imu=self.imu,
            timestep=32,
        )

        pose = self.manager.update(
            left_encoder_position=0.0,
            right_encoder_position=0.0,
        )

        self.assertIsNotNone(
            pose
        )

        self.assertAlmostEqual(
            pose.x,
            2.0,
        )

        self.assertAlmostEqual(
            pose.y,
            0.0,
        )

    def test_manager_uses_encoder_motion(self):
        self.manager.initialize(
            gps=self.gps,
            imu=self.imu,
            timestep=32,
        )

        first_pose = self.manager.update(
            left_encoder_position=0.0,
            right_encoder_position=0.0,
        )

        second_pose = self.manager.update(
            left_encoder_position=1.0,
            right_encoder_position=1.0,
        )

        self.assertIsNotNone(
            first_pose
        )

        self.assertIsNotNone(
            second_pose
        )

        self.assertNotEqual(
            second_pose.x,
            first_pose.x,
        )

    def test_manager_heading(self):
        self.manager.initialize(
            gps=self.gps,
            imu=self.imu,
            timestep=32,
        )

        pose = self.manager.update(
            left_encoder_position=0.0,
            right_encoder_position=0.0,
        )

        self.assertAlmostEqual(
            pose.heading,
            0.5,
        )
import unittest

from perception.camera import Camera


class FakeCamera(Camera):

    def __init__(self):
        self.initialized = False
        self.closed = False
        self.frame = "fake_frame"

    def initialize(self):
        self.initialized = True

    def get_frame(self):
        return self.frame

    def close(self):
        self.closed = True


class TestCameraInterface(unittest.TestCase):

    def setUp(self):
        self.camera = FakeCamera()

    def test_camera_can_be_initialized(self):
        self.camera.initialize()

        self.assertTrue(
            self.camera.initialized
        )

    def test_camera_returns_frame(self):
        self.camera.initialize()

        frame = self.camera.get_frame()

        self.assertEqual(
            frame,
            "fake_frame",
        )

    def test_camera_can_be_closed(self):
        self.camera.initialize()
        self.camera.close()

        self.assertTrue(
            self.camera.closed
        )

    def test_camera_lifecycle(self):
        self.assertFalse(
            self.camera.initialized
        )

        self.camera.initialize()

        self.assertTrue(
            self.camera.initialized
        )

        frame = self.camera.get_frame()

        self.assertEqual(
            frame,
            "fake_frame",
        )

        self.camera.close()

        self.assertTrue(
            self.camera.closed
        )


if __name__ == "__main__":
    unittest.main()
import sys
import unittest
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add project root so packages such as ai can be imported.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add raspberry_pi/src so localization, navigation and safety
# packages can be imported.
SRC_ROOT = PROJECT_ROOT / "raspberry_pi" / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def run_tests():
    tests_dir = Path(__file__).resolve().parent

    loader = unittest.TestLoader()

    suite = loader.discover(
        start_dir=str(tests_dir),
        pattern="test_*.py",
    )

    runner = unittest.TextTestRunner(
        verbosity=2,
    )

    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()

    if success:
        print("\n========================================")
        print("ALL SIMULATION CODE TESTS: PASS")
        print("========================================")
        sys.exit(0)

    print("\n========================================")
    print("SIMULATION CODE TESTS: FAIL")
    print("========================================")
    sys.exit(1)
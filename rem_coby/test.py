import pytest
from inspect_fiber import FiberInspector
from image_io import ImageInputter, ImageOutputter

# Test cases for pass/fail results of fiber optic inspection
tests = {
    '00': False,  # Expected result for input_00.png
    '01': False,  # Expected result for input_01.png
    '02': True,   # Expected result for input_02.png
    '03': True    # Expected result for input_03.png
}

@pytest.mark.parametrize("image_number, expected_result", tests.items())
def test_fiber_inspection_pass_fail(image_number: str, expected_result: bool) -> None:
    """Test if the fiber optic endface inspection passes or fails for 4 test images.

    The test passes if the pass or fail result matches the expected pass or fail result for each test image from the IEC 61300-3-35 standard.

    The test does not compare the number and size of the defects, only if the defects result in a pass or fail.

    The test also does not consider scratches as this feature is a work in progress.

    Args:
        image_number: The number of the test image taken from the key of each test case. This is concatenated with a hardcoded file path.
        expected_result: The expected pass/fail result for the test image.
    """

    # Input
    input_image_path = f"input/image_{image_number}.png"
    
    # Process
    inspector = FiberInspector(input_image_path)
    _, actual_pass, _ = inspector.inspect_fiber()

    # Output
    print(f"\nExpected pass/fail result for {input_image_path}: {expected_result}")
    print(f"Actual pass/fail result for {input_image_path}: {actual_pass}")

    # Test
    assert actual_pass == expected_result, f"Failed on image {input_image_path}"
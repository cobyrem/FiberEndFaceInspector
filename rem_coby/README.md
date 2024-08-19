# Coby Rem JHU Image Processing Final Project - Fiber Optic Endface Automated Inspection

## Required Libraries
- `argparse`: Parse input arguments.
- `typing`: Include additional datatypes for type hinting.
- `os`: Used for file path and name manipulation.
- `cv2`: OpenCV for image processing.
- `image_io`: Python module for importing and outputting images.
- `pytest`: Testing framework for python to quickly run code on test images.

## How to run
- Install Python. This code was only tested on Python 3.9.0
- Install libraries
- Open a terminal window
- Example command to run: `python3 inspect_fiber.py -i "input/image_##.png" -o "output/image_##_inspected"` 
	- Image locations can be modified to valid file locations
    - Pres the 'q' key to go through all intermittent images.
    - Comment out the scratch detection as needed on line 90
- Use the following command to run all tests at once using pytest: `clear & pytest test.py -vv -s`
	- To save the logs to a file called `log.txt` run the following command: `clear & pytest test.py -vv -s > log.txt`
- For ease of use all python scripts and input files should be in the same folder

## Files
- `input`: Folder that contains input test images.
- `output`: Folder that contains final output from inspecting the fiber optic endface images in `input`
- `truth`: Screenshots from IEC-61300-3-35 standard of highlighted defects, scratches, and region of images in `input`. Also contains if the fiber passed or failed which is used in `test.py`.
- `inspect_fiber.py`: Python code to inspect the fiber optic cable endface images in `input` and output to `output`.
- `image_io.py`: Python module for importing and outputting images.
- `test.py`: Python code using pytest to test images in `input` to the pass and fail statements in `truth`.
- `log.txt`: Output logs of `test.py`

## Comments
- Scratch detection is not integrated into final pass fail criteria due to need for improvement.
- See the sample terminal outputs below:
cobyarem@Cobys-MBP rem_coby % python3 inspect_fiber.py -i "input/image_00.png" -o "output/image_00"
Fiber passed inspection?: Failed
Fiber defects and scratches detected in each region: {'core': {'scratches': 0, 'defects': 0}, 'cladding': {'scratches': {'le_3um': 0, 'g_3um': 0}, 'defects': {'l_2um': 1, 'ge_2um_le_5um': 1, 'g_5um': 1}}, 'adhesive': {'scratches': 0, 'defects': 0}, 'contact': {'scratches': 0, 'defects': {'ge_10um': 0}}}
Modified image saved to output/image_00_inspected
cobyarem@Cobys-MBP rem_coby % python3 inspect_fiber.py -i "input/image_01.png" -o "output/image_01"
Fiber passed inspection?: Failed
Fiber defects and scratches detected in each region: {'core': {'scratches': 0, 'defects': 1}, 'cladding': {'scratches': {'le_3um': 0, 'g_3um': 0}, 'defects': {'l_2um': 0, 'ge_2um_le_5um': 0, 'g_5um': 0}}, 'adhesive': {'scratches': 0, 'defects': 0}, 'contact': {'scratches': 0, 'defects': {'ge_10um': 0}}}
Modified image saved to output/image_01_inspected
cobyarem@Cobys-MBP rem_coby % python3 inspect_fiber.py -i "input/image_02.png" -o "output/image_02"
Fiber passed inspection?: Passed
Fiber defects and scratches detected in each region: {'core': {'scratches': 0, 'defects': 0}, 'cladding': {'scratches': {'le_3um': 0, 'g_3um': 0}, 'defects': {'l_2um': 0, 'ge_2um_le_5um': 2, 'g_5um': 0}}, 'adhesive': {'scratches': 0, 'defects': 0}, 'contact': {'scratches': 0, 'defects': {'ge_10um': 0}}}
Modified image saved to output/image_02_inspected
cobyarem@Cobys-MBP rem_coby % python3 inspect_fiber.py -i "input/image_03.png" -o "output/image_03"
Fiber passed inspection?: Passed
Fiber defects and scratches detected in each region: {'core': {'scratches': 0, 'defects': 0}, 'cladding': {'scratches': {'le_3um': 0, 'g_3um': 0}, 'defects': {'l_2um': 1, 'ge_2um_le_5um': 3, 'g_5um': 0}}, 'adhesive': {'scratches': 0, 'defects': 0}, 'contact': {'scratches': 0, 'defects': {'ge_10um': 0}}}
Modified image saved to output/image_03_inspected

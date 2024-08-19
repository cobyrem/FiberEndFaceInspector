import argparse
from typing import Tuple, Dict
import cv2
from matplotlib import pyplot as plt
import numpy as np
from image_io import ImageInputter, ImageOutputter

class FiberInspector:
    def __init__(self, input_dir: str) -> None:
        """Load the image, default if fiber passed to False, and then mark the number of defects and scratches per region to zero.

        Args:
            input_dir: Path to the folder of input image files.
        """
        self.input_dir = input_dir
        self.center_x = 0
        self.center_y = 0
        self.center = (self.center_x, self.center_y)
        self.adhesive_radius = 0
        self.cladding_radius = 0
        self.contact_radius = 0
        self.core_radius = 0
        self.fiber = self.load_image()
        self.fiber_with_circles = self.fiber.copy()
        self.fiber_passed = False
        self.fiber_defects_scratches = {
            "core": {
                "scratches": 0,
                "defects": 0
            },
            "cladding": {
                "scratches": {
                    "le_3um": 0,
                    "g_3um": 0
                },
                "defects": {
                    "l_2um": 0,
                    "ge_2um_le_5um": 0,
                    "g_5um": 0
                }
            },
            "adhesive": {
                "scratches": 0,
                "defects": 0
            },
            "contact": {
                "scratches": 0,
                "defects": {
                    "ge_10um": 0
                }
            }
        }

    def load_image(self) -> np.ndarray:
        """Load the original image, create an image to create masks, and copy the image for the final color image.

        Returns:
            The original image specified in the command line argument.
        """
        self.inputter = ImageInputter(self.input_dir)
        image = self.inputter.image

        return image
    
    def show_image(self, image: np.ndarray, title: str) -> None:
        """Display an image with a title.

        Args:
            image: Image to display.
            title: The title of the image.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def inspect_fiber(self) -> Tuple[np.ndarray, Dict, bool]:
        """High level function to find and count the defects in each region and a prototype of detecting scratches on fiber optic endfaces.
        
        The final image with each region color coded, the defects color coded to each region, and the largest scratch in pink are displayed.

        The fiber optic enface is tested against IEC 61300-3-35 but only for defects as the scratch detection is a work in progress (See final paper for more details).
        
        Returns:
            The image with color coded regions/defects, and the largest detected scratch. The cable is also marked pass or fail and the dictionary of scratches and defects is also returned.
        """
        self.inspect_defects()
        # Comment out scratch inspection as needed
        self.inspect_scratches()
        self.show_image(self.fiber_with_circles,"8 Inspected")
        self.check_pass_fail()

        return self.fiber_with_circles, self.fiber_passed, self.fiber_defects_scratches

    def inspect_defects(self):
        """Inspect the fiber optic enface for defects.

        The image is first manipulated to enhance defects and reduce potential noise using gaussian blurs, filtering, thresholding, and morphological techniques.

        Next the main body of the fiber optic endace, the adhesive region and all inner regions, is isolated, and the other regions are also isolated based on their ratios to the adhesive region.
        
        Additional noise and artifacts from the low quality image are removed.

        Finally the defects in each region are detected, counted, and highlighted on top of the original image.
        """
        # Convert original image to gray scale
        fiber_gray = cv2.cvtColor(self.fiber, cv2.COLOR_BGR2GRAY)
        self.show_image(fiber_gray, "0 Gray Image")

        # Apply Gaussian blur to gray image
        blurred = cv2.GaussianBlur(fiber_gray, (5, 5), 0)
        self.show_image(blurred, "1 Blurred Image")

        # Apply Min-Max Ranking Filtering based on Mei et al. research
        min_filtered = cv2.erode(blurred, np.ones((3, 3), np.uint8),3)
        max_filtered = cv2.dilate(blurred, np.ones((3, 3), np.uint8),4)
        residual = max_filtered - min_filtered
        self.show_image(residual, "2 Residual Image")
        
        # Apply threshold segmentation based on Mei at al. research
        mean_val = np.mean(residual)
        std_val = np.std(residual)
        gamma = 1.8
        threshold_value = mean_val + gamma * std_val
        _, segmented = cv2.threshold(residual, threshold_value, 255, cv2.THRESH_BINARY)
        self.show_image(segmented, "3 Segmented Image")

        # Apply morphological erosion and dilation to remove small noise while trying to maintain similar defect sizes.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_segmented = cv2.erode(segmented, kernel, iterations=1)
        cleaned_segmented = cv2.dilate(cleaned_segmented, kernel, iterations=3)
        cleaned_segmented = cv2.erode(cleaned_segmented, kernel, iterations=1)
        self.show_image(cleaned_segmented, "4 Cleaned Segmented Image")

        # Find contours on the segmented image
        contours, _ = cv2.findContours(cleaned_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour which will likely be the adhesive region, unless something is blatantly wrong with the image.
        # Can be improved with more robust error checking and criteria. Perhaps evaluated the roundness and size of the largest contour.
        largest_contour = max(contours, key=cv2.contourArea)

        # Determine center and radii of the fiber optic endface regions
        (self.center_x, self.center_y), radius = cv2.minEnclosingCircle(largest_contour)
        self.center = (int(self.center_x), int(self.center_y))
        self.adhesive_radius = radius - 10
        self.cladding_radius = self.adhesive_radius - 10
        self.contact_radius = self.cladding_radius * 2
        self.core_radius = self.cladding_radius * 0.2

        # Remove additional noise and artifacts from cleaned image including the outer and inner rings of the adhesive and cladding regions.
        # Can be improved to better defect defects on inner and outer rings.
        contours = self.remove_fiber_artifacts(cleaned_segmented)

        # Detect and color code defects in each region 
        self.detect_defects(contours)

    def remove_fiber_artifacts(self, binary_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove artifacts from the adhesive and cladding region boundaries.

        Args:
            binary_image: Masked and threhsold fiber optic endface with majority of noise removed.

        Returns:
            The binary image with the adhesive and cladding region boundaries artifacts removed as well as all of the contours.
        """
        # Create masks of the adhesive and cladding region boundaries with some tolerance due to noisy input images.
        adhesive_mask = np.zeros_like(binary_image)
        cladding_mask = np.zeros_like(binary_image)
        center = (int(self.center_x), int(self.center_y))
        cv2.circle(adhesive_mask, center, int(self.adhesive_radius) + 50, (255), thickness=-1)
        cv2.circle(cladding_mask, center, int(self.cladding_radius) - 15, (255), thickness=-1)
        
        # Remove pixels between cladding and adhesive regions by making them black.
        inner_circle_mask = adhesive_mask - cladding_mask
        binary_image[inner_circle_mask > 0] = 0
        self.show_image(binary_image, "5 Remove outter region artifacts")

        # Detect all contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Identify and remove all core artifacts that may be contours.
        min_radius_threshold = 25
        inner_circle_contours = []
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            distance = np.sqrt((center[0] - self.center_x) ** 2 + (center[1] - self.center_y) ** 2)

            if distance < min_radius_threshold:
                inner_circle_contours.append(contour)

        # Remove the core contours from the binary image
        for contour in inner_circle_contours:
            cv2.drawContours(binary_image, [contour], -1, (0, 0, 0), thickness=-1)
        self.show_image(binary_image, "6 Removed core region artifacts")

        # Remove the core contours from the list of contours
        inner_circle_contours_tuples = [tuple(map(tuple, contour.reshape(-1, 2))) for contour in inner_circle_contours]
        filtered_contours = [contour for contour in contours if tuple(map(tuple, contour.reshape(-1, 2))) not in inner_circle_contours_tuples]

        return filtered_contours

    def detect_defects(self, contours: np.ndarray) -> None:
        """Detect, count, and color code all defects in all regions of the fiber optic endface and overlay them on the original image.

        Args:
            contours: All non-artifact contours detected in the fiber optic endface.
        """
        # Assign red, green, blue, and yellow to the core, cladding, adhesive, and contact regions respectivelly.
        core_color = (255, 0, 0)
        cladding_color = (0, 255, 0)
        adhesive_color = (0, 0, 255)
        contact_color = (255, 255, 0)

        # Draw colored circles for each region
        cv2.circle(self.fiber_with_circles, self.center, int(self.core_radius), core_color, 2)
        cv2.circle(self.fiber_with_circles, self.center, int(self.cladding_radius), cladding_color, 2)
        cv2.circle(self.fiber_with_circles, self.center, int(self.adhesive_radius), adhesive_color, 2)
        cv2.circle(self.fiber_with_circles, self.center, int(self.contact_radius), contact_color, 2)

        # Create an overlay for the defects
        highlighted_defects = self.fiber_with_circles.copy()

        for contour in contours:
            # Get the center and radius of the smallest enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            # Get the diameter of the smallest enclosing circle
            defect_size = 2 * radius  
            # Get the distance from middle of defect to center of fiber optic cable minus the radius.
            distance = (np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)) - radius

            # Check if the contour is within the core region
            if distance <= self.core_radius:
                self.fiber_defects_scratches['core']['defects'] += 1
                cv2.drawContours(highlighted_defects, [contour], -1, core_color, -1)

            # Check if the contour is within the cladding region. Multiplied factor for defect size accounts for morpology applied. Can be improved.
            elif self.core_radius < distance <= self.cladding_radius:
                if defect_size < 2*7:
                    self.fiber_defects_scratches['cladding']['defects']['l_2um'] += 1
                elif 2*7 <= defect_size <= 5*7:
                    self.fiber_defects_scratches['cladding']['defects']['ge_2um_le_5um'] += 1
                elif defect_size > 5*7:
                    self.fiber_defects_scratches['cladding']['defects']['g_5um'] += 1
                cv2.drawContours(highlighted_defects, [contour], -1, cladding_color, -1)

            # Check if the contour is within the adhesive region
            elif self.cladding_radius < distance <= self.adhesive_radius:
                self.fiber_defects_scratches['adhesive']['defects'] += 1
                cv2.drawContours(highlighted_defects, [contour], -1, adhesive_color, -1)

            # Check if the contour is within the contact region
            elif self.adhesive_radius < distance <= self.contact_radius:
                if defect_size >= 10*7:
                    self.fiber_defects_scratches['contact']['defects']['ge_10um'] += 1
                cv2.drawContours(highlighted_defects, [contour], -1, contact_color, -1)

        # Highlight the defects on the original image.
        self.fiber_with_circles = cv2.addWeighted(highlighted_defects, 0.5, self.fiber_with_circles, 1 - 0.5, 0)

    def check_pass_fail(self) -> None:
        """Determine if the fiber passes inspection based on defect counts.

        Scratches not used in pass fail criteria in first iteration of code.
        """
        core_defects = self.fiber_defects_scratches['core']['defects']
        cladding_defects = self.fiber_defects_scratches['cladding']['defects']
        contact_defects = self.fiber_defects_scratches['contact']['defects']

        if core_defects == 0 and cladding_defects['ge_2um_le_5um'] <= 5 and cladding_defects['g_5um'] == 0 and contact_defects['ge_10um'] == 0:
            self.fiber_passed = True
        else:
            self.fiber_passed = False

    def inspect_scratches(self):
        """Proof of concept to find and draw the largest scratch in an image.
        """

        # Black out the contact region.
        fiber_gray = cv2.cvtColor(self.fiber, cv2.COLOR_BGR2GRAY)
        height, width = fiber_gray.shape
        mask = np.zeros_like(fiber_gray)
        cv2.circle(mask, self.center, int(self.cladding_radius)-15, 255, thickness=-1)
        masked_gray = cv2.bitwise_and(fiber_gray, fiber_gray, mask=mask)

        # Increase the contrast on the image while limiting noise amplification using CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
        clahe_chosen = clahe.apply(masked_gray)

        # Apply Gaussian blur to reduce noise
        blurred_image_chosen = cv2.GaussianBlur(clahe_chosen, (5, 5), 0)

        # Create an empty color image for visualization (all black)
        binary_mask = np.zeros((height, width, 3), dtype=np.uint8)

        """Make each pixel white if the pixels 20 pixels to the left and right are darker.
        Note that 20 pixels to the right and left of the image are not evaluated. 
        This is a small section of the contact zone, and can be improved in future version 
        by only looking at pixels to the left on the right hand side of the image and vice versa.
        """
        for y in range(height):
            for x in range(20, width - 20):
                current_pixel = blurred_image_chosen[y, x]
                left_pixel = blurred_image_chosen[y, x - 20]
                right_pixel = blurred_image_chosen[y, x + 20]
                
                # Check if the current pixel is lighter than its neighbors.
                if current_pixel > left_pixel and current_pixel > right_pixel:
                    binary_mask[y, x] = [255, 255, 255]
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        self.show_image(binary_mask,"7 Mask of lighter pixels that may be lines")

        # Find the line that intersects through the most pixels 
        best_line = self.find_best_line_with_exact_stop(binary_mask)

        # Draw a contour around the line in pink on the image with color coded fiber defects.
        if best_line is not None:
            contour = np.array(best_line, dtype=np.int32).reshape((-1, 1, 2))            
            cv2.drawContours(
                self.fiber_with_circles, 
                [contour], 
                contourIdx=-1, 
                color=(255, 192, 203), 
                thickness=2
            )
        cv2.addWeighted(self.fiber_with_circles, 0.5, self.fiber_with_circles, 0.5, 0, self.fiber_with_circles)

    def find_best_line_with_exact_stop(self,binary_image):
        """Use the Probabilistic Hough Transform to detect lines with lower thresholds to get more lines
        
        A line is detected with 1 pixel granularity, at 1 degree angles, and has to be 30 pixels long, and can have up to 5 black pixels between white pixels.
        
        A vector for a line is calculated and the line with the most pixel intersections is returned.

        Args:
            binary_image: Mask of the pixels that can be potential lines.

        Returns:
            Corners for the best line/rectangle.
        """
        max_intersections = 0
        best_line = None

        lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=5)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:

                    # Calculate the line vector and its length
                    line_vec = np.array([x2 - x1, y2 - y1])
                    line_length = np.linalg.norm(line_vec)
                    if line_length == 0:
                        continue

                    # Calculate the width vector for the line.
                    width_vec = np.array([-line_vec[1], line_vec[0]]) / line_length * 4

                    # Get the four corners of the line/rectangle.
                    corners = [
                        (x1 + width_vec[0], y1 + width_vec[1]),
                        (x1 - width_vec[0], y1 - width_vec[1]),
                        (x2 + width_vec[0], y2 + width_vec[1]),
                        (x2 - width_vec[0], y2 - width_vec[1])
                    ]

                    # Create a mask of the line/rectangle.
                    line_mask = np.zeros_like(binary_image)
                    cv2.fillConvexPoly(line_mask, np.array(corners, dtype=np.int32), 1)

                    # Count the number of intersections
                    num_intersections = np.sum(binary_image & line_mask)
                    if num_intersections > max_intersections:
                        max_intersections = num_intersections
                        best_line = corners

        return best_line

def main(input_dir: str, output_dir: str) -> None:
    """Inspect the fiber cable to defects and scratches in different regions according to IEC 61300-3-35

    Args:
        input_dir: Path to the folder of input image files.
        output_dir: Path to the folder of output image files.
    """
    inspector = FiberInspector(input_dir)
    inspected_fiber, fiber_passed, fiber_defects_scratches  = inspector.inspect_fiber()

    if fiber_passed == True:
        print(f'Fiber passed inspection?: Passed')
    else:
        print(f'Fiber passed inspection?: Failed')

    print(f'Fiber defects and scratches detected in each region: {fiber_defects_scratches}')

    outputter = ImageOutputter(inspected_fiber)
    outputter.save_image(output_dir, '_inspected')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect fiber optic endface connectors')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the folder of input image files.')
    parser.add_argument('--output', '-o', type=str, required=True, help='Path to the folder of output image files.')

    args = parser.parse_args()

    main(args.input, args.output)
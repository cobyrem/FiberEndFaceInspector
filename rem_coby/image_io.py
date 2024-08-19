import os
import cv2

class ImageInputter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self.load_image()

    def load_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Error: unable to open {self.image_path}")
        return image
    
class ImageOutputter:
    def __init__(self, image):
        self.image = image
        self.new_image_name = None

    def create_output_image_name(self, original_image_path, text_to_append):
        original_image_name, extension = os.path.splitext(original_image_path)
        self.new_image_name = f"{original_image_name}{text_to_append}{extension}"

    def display_image(self, original_image_path, text_to_append):
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.create_output_image_name(original_image_path, text_to_append)
        cv2.imshow(self.new_image_name, rgb_image)
        print(f"++++++++++ Press any key to close image ++++++++++")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, original_image_path, text_to_append):
        self.create_output_image_name(original_image_path, text_to_append)
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{self.new_image_name}.png", rgb_image)
        print(f"Modified image saved to {self.new_image_name}")

    def display_and_save_image(self, original_image_path, text_to_append):
        self.display_image(original_image_path, text_to_append)
        self.save_image(original_image_path, text_to_append)
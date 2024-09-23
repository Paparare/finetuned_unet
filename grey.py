from PIL import Image
import os

test_directory = 'pic/output/'

for filename in os.listdir(test_directory):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Adjust file types as necessary
        image_path = os.path.join(test_directory, filename)

        with Image.open(image_path) as img:
            gray_image = img.convert('L')
            gray_image.save(image_path)

print("All images in the test folder have been converted to grayscale.")

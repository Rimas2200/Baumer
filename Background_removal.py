import os
from rembg import remove

input_folder = 'SIFT_ORM_SURF/b'
output_folder = 'output_images3/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, 'rb') as input_file:
            input_image = input_file.read()

        output_image = remove(input_image)

        with open(output_path, 'wb') as output_file:
            output_file.write(output_image)

        # image = Image.open(io.BytesIO(output_image))
        # image.show()

print("Обработка завершена!")



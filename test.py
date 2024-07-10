from PIL import Image 
from keras.models import load_model
import numpy as np

model2 = load_model("model.h5")

image_path = '2.png'
image = Image.open(image_path)
image = image.resize((150, 150))
image_array = np.array(image)


if image_array.shape[-1] == 4:  # to remove alpha channel
    image_array = image_array[:, :, :3]

image_array = np.expand_dims(image_array, axis=0)

print(image_array.shape)
print(image_array.dtype)

result = model2.predict(image_array)

if result[0][0] == 1:
    prediction = "ADHD"
else:
    prediction = "No ADHD"

print(prediction)

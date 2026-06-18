import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from four_or_nine import NeuralNetwork
from scipy.ndimage import center_of_mass, shift


data = np.load("mnist_4or9_model.npz")

nn = NeuralNetwork(
    input_size=784,
    hidden_size=32,
    output_size=1
)

nn.W1 = data["W1"]
nn.b1 = data["b1"]
nn.W2 = data["W2"]
nn.b2 = data["b2"]
image = Image.open("C:/Users/Theha/OneDrive/Pictures/four2.png").convert("L")
image_array = np.array(image)

rows = np.where(np.min(image_array, axis=1) < 250)[0]
cols = np.where(np.min(image_array, axis=0) < 250)[0]

if len(rows) > 0 and len(cols) > 0:
    image_array = image_array[
        rows.min():rows.max()+1,
        cols.min():cols.max()+1
    ]

image = Image.fromarray(image_array)
image = image.resize((28, 28), Image.Resampling.LANCZOS)
image_array = np.array(image)

image_array = 255 - image_array
image_array = image_array / 255.0

cy, cx = center_of_mass(image_array)

shift_y = 13.5 - cy
shift_x = 13.5 - cx

image_array = shift(image_array, [shift_y, shift_x])

image_array = image_array.reshape(1, 784)

prediction = nn.forward(image_array)
print(image_array.shape)

print("Raw prediction:", prediction)

if prediction[0][0] > 0.5:
    print("Prediction: 9")
else:
    print("Prediction: 4")

print("Min pixel:", image_array.min())
print("Max pixel:", image_array.max())

plt.imshow(image_array.reshape(28, 28), cmap='gray')
plt.colorbar()
plt.show()

plt.hist(image_array.flatten())
plt.show()

plt.show()

print(image_array.shape)
print(image_array.min())
print(image_array.max())
print(np.mean(image_array))

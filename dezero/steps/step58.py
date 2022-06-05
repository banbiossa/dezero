import numpy as np

from dezero.models import VGG16

model = VGG16(pretrained=True)
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
model.plot(x)


from PIL import Image

import dezero.core
import dezero.datasets
import dezero.utils

url = (
    "https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg"
)
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
img.show()

x = VGG16.preprocess(img)
print(type(x), x.shape)
x = x[np.newaxis]

model = VGG16(pretrained=True)
with dezero.core.test_mode():
    y = model(x)

predict_id = np.argmax(y.data)
model.plot(x, to_file="vgg.pdf")
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])

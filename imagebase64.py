import base64
import os

image = open("/Users/jminding/Documents/Research/Star Images 2/O/Alnitak.jpg", "rb")
image_read = image.read()
image_64_encode = base64.b64encode(image_read)
image_src = "data:image/jpeg;base64," + image_64_encode.decode('utf-8')
print(image_src)
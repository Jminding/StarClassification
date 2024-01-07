import albumentations as A
import cv2
import os
import matplotlib.pyplot as plt

transformations = [
    [[A.HorizontalFlip(p=1)], "_horizontalflipped"],
    [[A.VerticalFlip(p=1)], "_verticalflipped"],
    [[A.HorizontalFlip(p=1), A.VerticalFlip(p=1)], "_originflipped"],
]

transformations_2 = [
    [[A.Rotate(limit=[90, 90], p=1)], "_90"],
    [[A.Rotate(limit=[180, 180], p=1)], "_180"],
    [[A.Rotate(limit=[270, 270], p=1)], "_270"],
]

star_dir = "/Users/jminding/Documents/Research/Star Images LC"
for root, dirs, files in os.walk(star_dir, topdown=False):
    for name in files:
        if name.endswith(".jpg"):
            print("Doing " + name)
            img = cv2.imread(os.path.join(root, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for transformation in transformations:
                transform = A.Compose(transformation[0])
                transformed = transform(image=img)
                transformed_image = transformed["image"]
                cv2.imwrite(os.path.join(root, name.split(".")[0] + transformation[1] + ".jpg"), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

for root, dirs, files in os.walk(star_dir, topdown=False):
    for name in files:
        if name.endswith(".jpg"):
            print("Doing " + name)
            img = cv2.imread(os.path.join(root, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for transformation in transformations_2:
                transform = A.Compose(transformation[0])
                transformed = transform(image=img)
                transformed_image = transformed["image"]
                cv2.imwrite(os.path.join(root, name.split(".")[0] + transformation[1] + ".jpg"), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
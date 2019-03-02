# Segmentation-Utils

Some semantic segmentation utility functions. I could not find a script offering these functions and wasn't sure how to correctly implement them.

I was inspired among others by https://github.com/GeorgeSeif/Semantic-Segmentation-Suite. 



## Usage

```python
# Load image
import cv2
path = './data/eyth_dataset/masks/vid4/frame25.png'
mask = cv2.imread(path, 1)
mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
print(mask.shape) # (224, 224, 3)

# Each RGB colour get replaced by a one hot encoded segmentation map
# here we only have black and white
x = one_hot_image(mask)
print(x.shape) # (224, 224, 2)

# 2D Matrix where each pixel represents the label its colour label
x = one_hot_image_to_label_image(x)
print(x.shape) # (224, 224)

# Convert label back to rgb image
x = label_image_to_image(x)
print(x.shape) # (224, 224, 3)
```


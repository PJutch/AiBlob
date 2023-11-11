from PIL import Image
import numpy


def split_asarray(img: Image.Image):
    red, green, blue = img.split()
    return numpy.asarray(red), numpy.asarray(green), numpy.asarray(blue)

def merge_fromarray(red: numpy.ndarray,  green: numpy.ndarray, blue: numpy.ndarray):
    red_img = Image.fromarray(red).convert("L")
    green_img = Image.fromarray(green).convert("L")
    blue_img = Image.fromarray(blue).convert("L")
    return Image.merge("RGB", (red_img, green_img, blue_img))

image: Image.Image = Image.open('faces/train/profile/p (1).png')

flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

image_red, image_green, image_blue = split_asarray(image)
flipped_red, flipped_green, flipped_blue = split_asarray(flipped)

red_diff = image_red - flipped_red
green_diff = image_green - flipped_green
blue_diff = image_blue - flipped_blue

diff = merge_fromarray(red_diff / 2 + 128, green_diff / 2 + 128, blue_diff / 2 + 128)
diff.show()

x = numpy.arange(image_red.shape[1]).reshape((1, image_red.shape[1])).repeat(image_red.shape[0], axis=0)
y = numpy.arange(image_red.shape[0]).reshape((image_red.shape[0], 1)).repeat(image_red.shape[1], axis=1)
score = (x - image_red.shape[1] / 2) ** 2 + (y - image_red.shape[0] / 2) ** 2
print(numpy.sum((red_diff ** 2 * green_diff ** 2 + blue_diff ** 2) * score))

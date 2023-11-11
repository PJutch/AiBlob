from PIL import Image
import numpy

print('file;color')
for i in range(1, 11):
    print(f'apple{i}.jpg', end=';')
    with Image.open(f'test/apple{i}.jpg') as image:
        image: Image.Image
        image.load()

        array = numpy.asarray(image)
        if numpy.sum(array[0, 0]) > 255 * 1.5:
            mean_color = array[(numpy.sum(array, axis=2) < 512)].mean(axis=0)
        else:
            mean_color = array[(numpy.sum(array, axis=2) > 512)].mean(axis=0)

        if mean_color[0] > mean_color[1]:
            if mean_color[1] > 128:
                if mean_color[2] > 128:
                    print('3')
                else:
                    print('2')
            else:
                print('4')
        else:
            print('1')

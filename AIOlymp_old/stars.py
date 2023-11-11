from PIL import Image, ImageFilter
import numpy


def convolve(array: numpy.ndarray, kernel: numpy.ndarray):
    convolved = numpy.zeros_like(array)
    for i in range(array.shape[0]):
        convolved[i] = numpy.convolve(array[i], kernel)[kernel.size // 2:-(kernel.size // 2)]
    for i in range(array.shape[1]):
        convolved[:, i] = numpy.convolve(convolved[:, i], kernel)[kernel.size // 2:-(kernel.size // 2)]
    return convolved


print('file,star_count')
for ix in range(10):
    image: Image.Image = Image.open(f'test4/star{ix + 1}.jpg')
    image.load()

    grey: numpy.ndarray = numpy.min(numpy.asarray(image, dtype='float64'), axis=2)

    # result = Image.fromarray(convolve(grey, numpy.array([-0.125, -0.125, 1.5, -0.125, -0.125])))
    # result = result.filter(ImageFilter.SMOOTH)

    THRESHOLD = 0.45 * 256
    black_white = (grey > THRESHOLD)

    # reduced = convolve(black_white, numpy.array((-1, -1, -1, 7, -1, -1, -1))) * black_white

    # Image.fromarray(reduced * 255).show()

    stars = 0
    i_indices = numpy.arange(black_white.shape[0])
    i_indices.resize((black_white.shape[0], 1, 1))
    i_indices = i_indices.repeat(black_white.shape[1], axis=1)
    j_indices = numpy.arange(black_white.shape[1])
    j_indices.resize((1, black_white.shape[1], 1))
    j_indices = j_indices.repeat(black_white.shape[0], axis=0)
    indices = numpy.concatenate([i_indices, j_indices], axis=2)

    conditions = black_white
    conditions.resize((conditions.shape[0], conditions.shape[1], 1))
    conditions = conditions.repeat(2, axis=2)

    queue = list(indices[conditions])

    while queue:
        i = queue.pop(0)
        j = queue.pop(0)

        if not black_white[i, j]:
            continue

        stars += 1
        for di, dj in [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]:
            if 0 < i + di < black_white.shape[0] and 0 < j + dj < black_white.shape[1]:
                if black_white[i + di, j + dj]:
                    queue.append(i + di)
                    queue.append(j + dj)
                black_white[i + di, j + dj] = False
    # stars = int(numpy.sum(reduced) / (5 if ix == 4 or ix == 7 else 1))
    print(f'star{ix + 1}.jpg,'f'{stars}')

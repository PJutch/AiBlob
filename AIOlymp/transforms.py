import torch
from torchvision import transforms, datasets
from torchvision.transforms import v2
from matplotlib import pyplot

mean = torch.Tensor([0.4902, 0.4732, 0.4374])
std = torch.Tensor([0.1834, 0.1803, 0.1797])

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomRotation(degrees=30),
    v2.RandomPerspective(),
    v2.RandomResizedCrop(size=(32, 32), antialias=True),
    # v2.Normalize(mean=mean, std=std)
])

data = datasets.CIFAR10('CIFAR10', download=True, transform=transform)

# mean = torch.Tensor([0, 0, 0])
# std = torch.Tensor([0, 0, 0])
# for image, _ in data:
#     for i in range(3):
#         std_, mean_ = torch.std_mean(image[i])
#         mean[i] += mean_
#         std[i] += std_
# mean /= len(data)
# std /= len(data)
#
# print(f'mean={mean}, std={std}')

to_show = 5
for image, label in data:
    image: torch.Tensor = image
    image = torch.transpose(image, 0, 1)
    image = torch.transpose(image, 1, 2)

    pyplot.imshow(image)
    pyplot.show()

    to_show -= 1
    if to_show <= 0:
        break

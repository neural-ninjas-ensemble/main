import torch
from torchvision.transforms import v2

class Task3Dataset(Dataset):
    def __init__(self, transform=None):

        self.data = []
        self.labels = []

        # self.transform = transform

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        data = self.data[index]
        label = self.labels[index]
        # if not self.transform is None:
        #     img = self.transform(img)
        # label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.ids)


def transform_img(img):
  # H, W = 32, 32
  # img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

  transforms = v2.Compose([
      v2.RandomResizedCrop(size=(224, 224), antialias=True),
      v2.RandomHorizontalFlip(p=0.5),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  img = transforms(img)

  return img


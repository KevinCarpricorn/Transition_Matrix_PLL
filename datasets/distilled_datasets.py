import torch.utils.data as Data
from PIL import Image

class distilled_dataset(Data.Dataset):
    def __init__(self, distilled_images, distilled_partial_labels, distilled_bayes_labels, transform=None,
                 target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.distilled_images = distilled_images
        self.distilled_partial_labels = distilled_partial_labels
        self.distilled_bayes_labels = distilled_bayes_labels

    def __getitem__(self, index):
        img, bayes_label, partial_label = self.distilled_images[index], self.distilled_bayes_labels[index], \
        self.distilled_partial_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            bayes_label, partial_label = self.target_transform(bayes_label), self.target_transform(partial_label)

        return img, bayes_label, partial_label, index

    def __len__(self):
        return len(self.distilled_images)
import torch

class Composition:
    def __init__(self, transforms, probability=1):
        self.__init__()
        self.transforms = transforms
        self.probability = probability

    def __call__(self, x):
        for t in self.transforms:
            x = t(x) if(torch.rand(1) < self.probability) else x
        return x

    def __len__(self):
        return len(self.transforms)


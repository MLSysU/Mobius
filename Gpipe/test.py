import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

class Model(torch.nn.Module):
    def __init__(self,layer):
        super(Model, self).__init__()
        self.fc = layer

    def my_try(self):
        Init(layer=torch.nn.Linear(1, 1))

model = Model(layer=torch.nn.Linear(1, 1))
model.my_try()



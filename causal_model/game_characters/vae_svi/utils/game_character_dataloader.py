import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader ,random_split
from torchvision import datasets ,models , transforms
from torch import nn
import torch.nn.functional as F
from functools import reduce

class GameCharacterFullData(Dataset):
  def __init__(self, transforms, root_path, mode):
    # Change the following path to a more generalizable form. Like download it from
    # github or something like that. Make it usable to anyone.
    self.root_path = root_path
    self.train_path = self.root_path + 'train/'
    self.test_path = self.root_path + 'test/'
    self.train_csv = self.train_path + 'train.csv'
    self.test_csv = self.test_path + 'test.csv'
    self.mode = mode
    self.train_df = pd.read_csv(self.train_csv)
    self.test_df = pd.read_csv(self.test_csv)
    self.transforms = transforms

  def __getitem__(self, idx):
    if self.mode == "train":
      d = self.train_df.iloc[idx]
      image = Image.open(self.train_path + d["img_name"] + ".png").convert("RGB")
    else:
      d = self.test_df.iloc[idx]
      image = Image.open(self.test_path + d["filename"] + ".png").convert("RGB")
    
    # Extracting only the action reaction labels, coz that's what we condition on.
    lbl = d[["actor_action_Attacking", "actor_action_Taunt", "actor_action_Walking", "reactor_action_Attacking", "reactor_action_Dying", "reactor_action_Hurt", "reactor_action_Idle"]]
    actor = (torch.tensor(d[["actor_name_Satyr", "actor_name_Golem"]].tolist(), dtype=torch.float32)!=0).nonzero().squeeze(1)[0]
    reactor = (torch.tensor(d[["reactor_name_Satyr", "reactor_name_Golem"]].tolist(), dtype=torch.float32)!=0).nonzero().squeeze(1)[0]

    actor_type = ((torch.tensor(d[["actor_type_satyr1", "actor_type_satyr2", "actor_type_satyr3", "actor_type_golem1", "actor_type_golem2", "actor_type_golem3"]].tolist(), dtype=torch.float32)!=0).nonzero().squeeze(1)[0]) % 3
    reactor_type = ((torch.tensor(d[["reactor_type_satyr1", "reactor_type_satyr2", "reactor_type_satyr3", "reactor_type_golem1", "reactor_type_golem2", "reactor_type_golem3"]].tolist(), dtype=torch.float32)!=0).nonzero().squeeze(1)[0]) % 3




    label = torch.tensor(lbl.tolist(), dtype=torch.float32)
    if self.transforms is not None:
        
        xp = self.transforms(image)
      # transform x to a linear tensor from bx * a1 * a2 * ... --> bs * A
        #xp_1d_size = reduce(lambda a, b: a * b, xp.size()[1:])

        #xp = xp.view(-1, xp_1d_size)
        #xp = xp.squeeze(0)
        assert not np.isnan(xp.sum())
    return xp, label, actor, reactor, actor_type, reactor_type

  def __len__(self):
    if self.mode == "train":
      return self.train_df.shape[0]
    else:
      return self.test_df.shape[0]

def setup_data_loaders(dataset, root_path, batch_size, transforms):
    train_dataset = dataset(transforms, root_path, mode="train")
    test_dataset = dataset(transforms, root_path, mode="test")    

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader


        


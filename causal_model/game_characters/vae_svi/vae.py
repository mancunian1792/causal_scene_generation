'''
Causal Variational Autoencoder - Probabilistic model for game character dataset.
'''
__author__ = 'Harish Ramani'
__email__ = 'ramani.h@northeastern.edu'


import argparse
import numpy as np
import torch
import torch.nn as nn
import visdom
import os
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from utils.game_character_dataloader import GameCharacterFullData, setup_data_loaders
from torchvision import transforms,models
import time

torch.set_default_tensor_type('torch.cuda.FloatTensor')


GDRIVE_DATA_PATH = "gdrive/My Drive/causal_scene_generation/vae_svi/data/"
LOCAL_DATA_PATH = "./data/"

GDRIVE_MODEL_PATH = "gdrive/My Drive/causal_scene_generation/vae_svi/model/"
LOCAL_MODEL_PATH = "./model/"


DATA_PATH = GDRIVE_DATA_PATH if 'COLAB_GPU' in os.environ else LOCAL_DATA_PATH
MODEL_PATH = GDRIVE_MODEL_PATH if 'COLAB_GPU' in os.environ else LOCAL_MODEL_PATH
values = {
      "action": ["Attacking", "Taunt", "Walking"],
      "reaction": ["Dying", "Hurt", "Idle", "Attacking"],
      "strength": ["Low", "High"],
      "defense": ["Low", "High"],
      "attack": ["Low", "High"],
      "actor": ["Satyr", "Golem"],
      "reactor": ["Satyr", "Golem"],
      "Satyr": ["satyr1", "satyr2", "satyr3"],
      "Golem": ["golem1", "golem2", "golem3"]
  }

cpts = {
    "action": torch.tensor([[[[0.1, 0.3, 0.6], [0.3, 0.5, 0.2]], [[0.3, 0.4, 0.4], [0.5,0.4,0.1]]],
                              [[[0.1, 0.2, 0.7], [0.4,0.3,0.3]], [[0.2, 0.4, 0.4], [0.6, 0.3, 0.1]]]]),
    
    "reaction": torch.tensor([[[[[0.5, 0.4, 0.05, 0.05], [0.2, 0.6, 0.1, 0.1], [0.001, 0.001, 0.997, 0.001]],
                    [[0.4, 0.3,0.1, 0.2], [0.1, 0.5, 0.2, 0.2], [0.001, 0.001, 0.99, 0.008]]],
                    [[[0.1, 0.3, 0.55, 0.05], [0.1, 0.2, 0.65, 0.05], [0.001, 0.001, 0.997, 0.001]],
                    [[0.3, 0.2, 0.3, 0.2],[0.1, 0.3, 0.4, 0.2],[0.001, 0.001, 0.99, 0.008]]]],
                  [[[[0.3, 0.3, 0.399, 0.001],[0.2, 0.4, 0.399, 0.001],[0.001, 0.001, 0.997, 0.001]],
                    [[0.3, 0.4, 0.1, 0.2],[0.3, 0.3, 0.1, 0.3],[0.001, 0.001, 0.99, 0.008]]],
                    [[[0.2, 0.3, 0.49, 0.01],[0.1, 0.2, 0.69, 0.01],[0.001, 0.001, 0.997, 0.001]],
                    [[0.2, 0.2, 0.4, 0.2],[0.1, 0.1, 0.4, 0.4],[0.001, 0.001, 0.99, 0.008]]]]]),
    "character": torch.tensor([0.5, 0.5]),
    "type": torch.tensor([[0.33, 0.34, 0.33], [0.33, 0.34, 0.33]]),
    "strength": torch.tensor([[[0.4, 0.6], [0.2, 0.8], [0.5, 0.5]], [[0.6, 0.4], [0.5, 0.5], [0.8, 0.2]]]),
    "defense": torch.tensor([[[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]], [[0.5, 0.5], [0.4, 0.6], [0.6, 0.4]]]),
    "attack": torch.tensor([[[0.2, 0.8], [0.6, 0.4], [0.8, 0.2]], [[0.75, 0.25], [0.4, 0.6], [0.9, 0.1]]])
}

inverse_cpts = {
    "reaction_strength": torch.tensor([[[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]], [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]], [[0.4, 0.6], [0.3, 0.7], [0.65, 0.35]]]),
    "reaction_defense": torch.tensor([[[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]], [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]], [[0.4, 0.6], [0.3, 0.7], [0.65, 0.35]]]),
    "reaction_attack": torch.tensor([[[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]], [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]], [[0.4, 0.6], [0.3, 0.7], [0.65, 0.35]]]),
    "action_strength": torch.tensor([[[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]], [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]]]),
    "action_defense": torch.tensor([[[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]], [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]]]),
    "action_attack": torch.tensor([[[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]], [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]]])
}

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=1024, num_labels=7):
        super().__init__()
        self.cnn = get_cnn_encoder(image_channels=3) # Currently this returns only for 1024 hidden dimensions. Need to change that
        # setup the two linear transformations used
        self.fc21 = nn.Linear(hidden_dim+num_labels, z_dim)
        self.fc22 = nn.Linear(hidden_dim+num_labels, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x,y):
        '''
        Here if i get an array of [xs, ys] what should i do ?
        xs is gonna be of the shape (32, 3, 400,400) and ys is gonna be of the shape (32,10)
        '''
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        #x = x.reshape(-1, 40000)
        # then compute the hidden units
        hidden = self.cnn(x)
        hidden = self.softplus(hidden) # This should return a [1, 1024] vector.
        # then return a mean vector and a (positive) square root covariance

        # each of size batch_size x z_dim
        hidden = torch.cat([hidden, y], dim=-1)
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


def get_seq_decoder(hidden_dim=1024, image_channels=3):
    return nn.Sequential(
            UnFlatten(), # (32, 1024, 1, 1)
            nn.ConvTranspose2d(hidden_dim, 512, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=13, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=11, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, image_channels, kernel_size=2, stride=1),
            nn.Sigmoid() # (32, 3, 400,400)
        )

def get_cnn_encoder(image_channels=3):
    return nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=2, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            Flatten()
        )




# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_labels=7):
        super().__init__()
        self.cnn_decoder = get_seq_decoder(hidden_dim, 3) # image_channels is 3
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim+num_labels, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc21 = nn.Linear(hidden_dim, 400)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z, y):
        # define the forward computation on the latent z
        # first compute the hidden units
        concat_z = torch.cat([z, y], dim=-1)
        hidden = self.softplus(self.fc1(concat_z))
        #hidden = self.softplus(self.fc2(hidden))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.cnn_decoder(hidden)
        return loc_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 500 hidden units
    def __init__(self, z_dim=128, hidden_dim=1024, use_cuda=False, num_labels=7):
        super().__init__()
        self.output_size = num_labels
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim, num_labels)
        self.decoder = Decoder(z_dim, hidden_dim, num_labels) # 3 channel image.

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x,y, actorObs, reactorObs, actor_typeObs, reactor_typeObs):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        options = dict(dtype=x.dtype, device=x.device)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            # The label y  is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)

            #print(f"In model actor is {actorObs}, reactor is {reactorObs}, actor_type is {actor_typeObs} and reactor_type is {reactor_typeObs}")
            '''
            Causal Model
            '''

            '''
            The below should basically be a concatenation of actor's action and reactor's reaction.
            '''

            actor = pyro.sample("actor", dist.Categorical(cpts["character"]), obs=actorObs).cuda()
            reactor = pyro.sample("reactor", dist.Categorical(cpts["character"]), obs=reactorObs).cuda()

            # To choose the type of Satyr or Golem (type 1, 2 or 3. This translates to different image of that character.)
            actor_type = pyro.sample("actor_type", dist.Categorical(cpts["type"][actor]), obs=actor_typeObs).cuda()
            reactor_type = pyro.sample("reactor_type", dist.Categorical(cpts["type"][reactor]), obs=reactor_typeObs).cuda()

            # To choose the strength, defense and attack based on the character and its type. Either Low or High
            actor_strength = pyro.sample("actor_strength", dist.Categorical(cpts["strength"][actor, actor_type])).cuda()
            actor_defense = pyro.sample("actor_defense", dist.Categorical(cpts["defense"][actor, actor_type])).cuda()
            actor_attack = pyro.sample("actor_attack", dist.Categorical(cpts["attack"][actor, actor_type])).cuda()

            # To choose the character's(actor, who starts the fight) action based on the strength, defense and attack capabilities
            actor_action = pyro.sample("actor_action", dist.OneHotCategorical(cpts["action"][actor_strength, actor_defense, actor_attack]), obs=y[..., :len(values["action"])].cuda()).cuda()

            # Converting onehot categorical to categorical value
            sampled_actor_action = actor_action[..., :].nonzero()[:, 1].cuda()
            # To choose the other character's strength, defense and attack based on the character and its type
            reactor_strength = pyro.sample("reactor_strength", dist.Categorical(cpts["strength"][reactor, reactor_type])).cuda()
            reactor_defense = pyro.sample("reactor_defense", dist.Categorical(cpts["defense"][reactor, reactor_type])).cuda()
            reactor_attack = pyro.sample("reactor_attack", dist.Categorical(cpts["attack"][reactor, reactor_type])).cuda()

            # To choose the character's (reactor, who reacts to the actor's action in a duel) reaction based on its own strength, defense , attack and the other character's action.
            reactor_reaction = pyro.sample("reactor_reaction", dist.OneHotCategorical(cpts["reaction"][reactor_strength, reactor_defense, reactor_attack, sampled_actor_action]), obs=y[..., len(values["action"]):].cuda()).cuda()

            ys = torch.cat([actor_action, reactor_reaction], dim=-1).cuda()
            '''
            Basically, the following should be a concatenation of actor's action and reactor's reaction
            '''
            #alpha_prior = torch.ones(x.shape[0], self.output_size, **options) / (1.0 * self.output_size)
            #ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)

            loc_img = self.decoder.forward(z,ys)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(3), obs=x)

            #print(f"actor is {actor},reactor is {reactor}, actor_type is {actor_type}, reactor_type is {reactor_type},actor_strength is {actor_strength}, actor_defense is {actor_defense},actor_attack is {actor_attack}, actor_action is {actor_action}, sampled_actor_action is {sampled_actor_action}, reactor_strength is {reactor_strength}, reactor_attack is {reactor_attack}, reactor_defense is {reactor_defense},reactor_reaction is {reactor_reaction}, ys is {ys}")
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, y, actorObs, reactorObs, actor_typeObs, reactor_typeObs):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            action, reaction = torch.nonzero(y[..., :3])[:, 1].cuda(), torch.nonzero(y[..., 3:])[:, 1].cuda()
            # use the encoder to get the parameters used to define q(z|x)
            actor_strength = pyro.sample("actor_strength", dist.Categorical(inverse_cpts["action_strength"][action, actor_typeObs])).cuda()
            actor_defense = pyro.sample("actor_defense", dist.Categorical(inverse_cpts["action_defense"][action, actor_typeObs])).cuda()
            actor_attack = pyro.sample("actor_attack", dist.Categorical(inverse_cpts["action_attack"][action, actor_typeObs])).cuda()

            reactor_strength = pyro.sample("reactor_strength", dist.Categorical(inverse_cpts["reaction_strength"][reaction, reactor_typeObs])).cuda()
            reactor_defense = pyro.sample("reactor_defense", dist.Categorical(inverse_cpts["reaction_defense"][reaction, reactor_typeObs])).cuda()
            reactor_attack = pyro.sample("reactor_attack", dist.Categorical(inverse_cpts["reaction_attack"][reaction, reactor_typeObs])).cuda()


            #print(f"actor is {actorObs}, reactor is {reactorObs}, actor_type is {actor_typeObs}, reactor_type is {reactor_typeObs}, actor_strength is {actor_strength}, actor_defense is {actor_defense}, actor_attack is {actor_attack}, actor_action is {action},reactor_strength is {reactor_strength},reactor_attack is {reactor_attack},reactor_defense is {reactor_defense}, reactor_reaction is {reaction}")

            z_loc, z_scale = self.encoder.forward(x,y) # y -> action and reaction
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
    
    def inference_model(self, x):
      # register PyTorch module `decoder` with Pyro
        options = dict(dtype=x.dtype, device=x.device)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            # The label y  is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)

            #print(f"In model actor is {actorObs}, reactor is {reactorObs}, actor_type is {actor_typeObs} and reactor_type is {reactor_typeObs}")
            '''
            Causal Model
            '''

            '''
            The below should basically be a concatenation of actor's action and reactor's reaction.
            '''

            actor = pyro.sample("actor", dist.Categorical(cpts["character"])).cuda()
            reactor = pyro.sample("reactor", dist.Categorical(cpts["character"])).cuda()

            # To choose the type of Satyr or Golem (type 1, 2 or 3. This translates to different image of that character.)
            actor_type = pyro.sample("actor_type", dist.Categorical(cpts["type"][actor])).cuda()
            reactor_type = pyro.sample("reactor_type", dist.Categorical(cpts["type"][reactor])).cuda()

            # To choose the strength, defense and attack based on the character and its type. Either Low or High
            actor_strength = pyro.sample("actor_strength", dist.Categorical(cpts["strength"][actor, actor_type])).cuda()
            actor_defense = pyro.sample("actor_defense", dist.Categorical(cpts["defense"][actor, actor_type])).cuda()
            actor_attack = pyro.sample("actor_attack", dist.Categorical(cpts["attack"][actor, actor_type])).cuda()

            # To choose the character's(actor, who starts the fight) action based on the strength, defense and attack capabilities
            actor_action = pyro.sample("actor_action", dist.OneHotCategorical(cpts["action"][actor_strength, actor_defense, actor_attack])).cuda()

            # Converting onehot categorical to categorical value
            sampled_actor_action = actor_action[..., :].nonzero()[:, 1].cuda()
            # To choose the other character's strength, defense and attack based on the character and its type
            reactor_strength = pyro.sample("reactor_strength", dist.Categorical(cpts["strength"][reactor, reactor_type])).cuda()
            reactor_defense = pyro.sample("reactor_defense", dist.Categorical(cpts["defense"][reactor, reactor_type])).cuda()
            reactor_attack = pyro.sample("reactor_attack", dist.Categorical(cpts["attack"][reactor, reactor_type])).cuda()

            # To choose the character's (reactor, who reacts to the actor's action in a duel) reaction based on its own strength, defense , attack and the other character's action.
            reactor_reaction = pyro.sample("reactor_reaction", dist.OneHotCategorical(cpts["reaction"][reactor_strength, reactor_defense, reactor_attack, sampled_actor_action])).cuda()

            ys = torch.cat([actor_action, reactor_reaction], dim=-1).cuda()
            '''
            Basically, the following should be a concatenation of actor's action and reactor's reaction
            '''
            #alpha_prior = torch.ones(x.shape[0], self.output_size, **options) / (1.0 * self.output_size)
            #ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)

            loc_img = self.decoder.forward(z,ys)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(3), obs=x)

            #print(f"actor is {actor},reactor is {reactor}, actor_type is {actor_type}, reactor_type is {reactor_type},actor_strength is {actor_strength}, actor_defense is {actor_defense},actor_attack is {actor_attack}, actor_action is {actor_action}, sampled_actor_action is {sampled_actor_action}, reactor_strength is {reactor_strength}, reactor_attack is {reactor_attack}, reactor_defense is {reactor_defense},reactor_reaction is {reactor_reaction}, ys is {ys}")
            # return the loc so we can visualize it later
            return loc_img
    
    def inference_guide(self, x):
      pass


    # define a helper function for reconstructing images
     # define a helper function for reconstructing images
    def reconstruct_img(self, x, y):
        # encode image x
        z_loc, z_scale = self.encoder(x,y)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z, y)
        return loc_img

def plot_vae_samples(vae):
    x = torch.zeros([1, 3, 400, 400])
    for i in range(2):
        images = []
        for rr in range(100):
            # get loc from the model
            sample_loc_i = vae.model(x.cuda(), y.cuda())
            img = sample_loc_i[0].view(3, 400, 400).permute(1,2,0).cpu().data.numpy()
            images.append(img)
    return images

def main(args):
    # clear param store
    pyro.clear_param_store()
    #pyro.enable_validation(True)

    # setup MNIST data loaders
    # train_loader, test_loader
    transform = transforms.Compose([
                                    transforms.Resize((400,400)),
                                    transforms.ToTensor()
                                ])

    train_loader, test_loader = setup_data_loaders(dataset=GameCharacterFullData, root_path = DATA_PATH, batch_size=32, transforms=transform)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda, num_labels = 7)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

   
     # setup visdom for visualization
    if args.visdom_flag:
        vis = visdom.Visdom(port='8097')

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, y, actor, reactor, actor_type, reactor_type in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
                actor = actor.cuda()
                reactor = reactor.cuda()
                actor_type = actor_type.cuda()
                reactor_type = reactor_type.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x,y, actor, reactor, actor_type, reactor_type)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for i, (x, y, actor, reactor, actor_type, reactor_type) in enumerate(test_loader):
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    actor = actor.cuda()
                    reactor = reactor.cuda()
                    actor_type = actor_type.cuda()
                    reactor_type = reactor_type.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x,y, actor, reactor, actor_type, reactor_type)
                # pick three random test images from the first mini-batch and
                # visualize how well we're reconstructing them
                if i == 0:
                    if args.visdom_flag:
                        plot_vae_samples(vae, vis)
                        reco_indices = np.random.randint(0, x.shape[0], 3)
                        for index in reco_indices:
                            test_img = x[index, :]
                            reco_img = vae.reconstruct_img(test_img)
                            vis.image(test_img.reshape(400, 400).detach().cpu().numpy(),
                                      opts={'caption': 'test image'})
                            vis.image(reco_img.reshape(400, 400).detach().cpu().numpy(),
                                      opts={'caption': 'reconstructed image'})
            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))
    
    return vae, optimizer


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.1')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=101, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    args = parser.parse_args()

    model, optimizer = main(args)
    t = time.time()
    torch.save(model.state_dict(), MODEL_PATH+"vae_model"+str(t)+".pkl")
    optimizer.save(MODEL_PATH+"optimizer"+str(t)+".pkl")
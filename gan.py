# from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os, pathlib
import random


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 1000

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# Frequency of printing training stats (in steps)
print_freq = 50

# Frequency of testing the generator (in steps)
test_freq = 500

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#https://discuss.pytorch.org/t/how-do-i-print-output-of-each-layer-in-sequential/5773/3
class PrintLayer(nn.Module):
    def __init__(self,f):
        self.f = f
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        if self.f:
            print(x.shape)
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu,f=False):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            PrintLayer(f),
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,1),
            PrintLayer(f),
            # state size. 8 x 127 x 127
            nn.Conv2d(8,16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,1),
            PrintLayer(f),
            # state size. 16 x 62 x 62
            nn.Conv2d(16,32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,1),
            PrintLayer(f),
            # state size. 32 x 30 x 30
            nn.Conv2d(32,64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,1),
            PrintLayer(f),
            # state size. 64 x 14 x 14
            nn.Conv2d(64,128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,1),
            PrintLayer(f),
            # state size. 128 x 6 x 6
            nn.Conv2d(128,256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,1),
            PrintLayer(f),
            # state size. 256 x 2 x 2
            nn.Flatten(),
            nn.Linear(1024,128, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            PrintLayer(f),
            # state size. 128
            nn.Linear(128,1, bias=False),
            nn.Sigmoid(),
            PrintLayer(f)
            # state size. 1
        )

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):

    def __init__(self, ngpu,f=False):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            PrintLayer(f),
            # input is 100 x 1 x 1, going into a convolution
            nn.ConvTranspose2d( nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            PrintLayer(f),
            # state size. (512) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            PrintLayer(f),
            # state size. (256) x 8 x 8
            nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            PrintLayer(f),
            # state size. (128) x 16 x 16
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            PrintLayer(f),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d( 64,32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            PrintLayer(f),
            # state size. 32 x 64 x 64
            nn.ConvTranspose2d( 32,16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            PrintLayer(f),
            # state size. 16 x 128 x 128
            nn.ConvTranspose2d( 16,8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            PrintLayer(f),
            # state size. 8 x 256 x 256
            nn.Conv2d( 8,nc, 3, 1, 1, bias=False),
            nn.Tanh(),
            PrintLayer(f)
            # state size. 1 x 256 x 256
        )
    def forward(self, input):
        return self.main(input)

def run():
    t = open("text.txt", "a")
    t.write("Now the file has more content!")
    t.close()
    # Set random seed for reproducibility
    # manualSeed = 999
    manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    path=str(pathlib.Path(__file__).parent)+"/eyes/"
    data=[]
    for f in os.listdir(path):
        # print(len(data))
        if f.endswith(".npz") and "fg_cut" in f:
            print(f)
            t = open("text.txt", "a")
            t.write(f)
            t.close()
            datapoint=np.load(path+f, allow_pickle=True)
            for x in datapoint['arr_0']:
                # print(len(data))
                # print(x.shape)
                if x.shape == (256,256) and np.count_nonzero(~np.isnan(x)) > 3000:
                    # print(x.shape)
                    data.append(torch.from_numpy(np.nan_to_num(x)))
    # print(len(data))
    random.shuffle(data)

    # for x in data:
    #     plt.imshow(x, cmap='gray', vmin=0, vmax=255)
    #     plt.show()
    
    data=torch.stack(data)
    data=data/(data.max()/2)-1
    dataset=TensorDataset(data)
    # # We can use an image folder dataset the way we have it setup.
    # # Create the dataset
    # dataset = dset.ImageFolder(root=dataroot,
    #                         transform=transforms.Compose([
    #                             transforms.Resize(image_size),
    #                             transforms.CenterCrop(image_size),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                         ]))
    # # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
                                            
    # # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    # real_batch = next(iter(dataloader))

   
    # for batch in dataloader:
    #     print("g")
    #     plt.figure(figsize=(8,8))
    #     plt.axis("off")
    #     plt.title("Training Images")
    #     plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    #     plt.show()

    # Create the generator
    netG = Generator(ngpu, True).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    # print(netG)

    # Create the Discriminator
    netD = Discriminator(ngpu, True).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    # print(netD)
     
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        t = open("text.txt", "a")
        t.write(str(epoch))
        t.close()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device).unsqueeze(1)
            b_size = real_cpu.size(0)
            # print(b_size)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = torch.squeeze(netD(real_cpu))
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            # print(fake.shape)
            # print(fake.shape)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if iters % print_freq == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % test_freq == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu().numpy()
                img_list.append(fake)
                np.save("produced.npy",img_list)

            iters += 1

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")

    PATH = "netG.pt"
    torch.save(netG, PATH)
    
    PATH = "netD.pt"
    torch.save(netD, PATH)
    
    # fig = plt.figure(figsize=(8,8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # HTML(ani.to_jshtml())
        


if __name__ == '__main__':
    print('starting')    
    run()

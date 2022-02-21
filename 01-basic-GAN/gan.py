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
import sys
import net_architectures



# Define CUDA env variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(torch.cuda.device_count())

# Parse command line arguments

parser = argparse.ArgumentParser(description='Run GAN With Cyclone Data')
parser.add_argument('format', metavar='format', type=str, nargs=1, choices=['A','B','C','D','E'], help='data format to use')
parser.add_argument('net', metavar='net', type=str, nargs=1, choices=["Large_Net", "Small_Net"], help='which network variation to load')

parser.add_argument('path', metavar='path', default="/project/ciid/projects/ML-TC/", type=str, nargs="?", help='the base path to use')
parser.add_argument('--vars', nargs="+", help="the set of variables to use")

args = parser.parse_args()
print(args)
# input()
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
ngpu = 1

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

def run():
    base_path=args.path
    data_path=base_path+"Data/"
    num=0
    for dir in [name for name in os.listdir(base_path)]:
        # print(dir)
        if dir.isnumeric():
            num=max(num,int(dir))
    save_path=base_path+str(num+1)+"/"
    os.mkdir(save_path)
    sys.stdout = open(save_path+'output.txt', 'w+')
    print(save_path)
    t = open(save_path+"text.txt", "a")
    t.write("Run starting")
    # Set random seed for reproducibility
    manualSeed = 357
    # manualSeed = random.randint(1, 10000) # use if you want new results
    t.write("Random Seed: "+str(manualSeed))
    t.close()
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # path=str(pathlib.Path(__file__).parent)+"/Data/"
    data=[]
    for f in os.listdir(data_path):
        # print(len(data))
        if f.endswith(".npz") and args.format[0] in f:
            t = open(save_path+"text.txt", "a")
            t.write(f)
            t.close()
            datapoint=np.load(data_path+f, allow_pickle=True)
            for x in datapoint['arr_0']:
                # print(len(data))
                # print(x.shape)
                s=x.shape
                if s.count(s[0]) == len(s) and np.count_nonzero(~np.isnan(x)) > 3000:
                    # print(x.shape)
                    data.append(torch.from_numpy(np.nan_to_num(x)))
    # print(len(data))
    random.shuffle(data)

    # for x in data:
    #     plt.imshow(x)
    #     plt.show()
    # input()
    data=torch.stack(data)
    data=data/(data.max()/2)-1
    dataset=TensorDataset(data)

    # # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
                                            
    # # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    #print(str(torch.cuda.is_available()))
    t = open(save_path+'text.txt', 'w+')
    t.write(str(torch.cuda.is_available()))
   # t.write(cudaRuntimeGetVersion())
   #t.write(str(torch.cuda.current_device()))
    t.close()
    # Get the right architecture class
    arch=eval('net_architectures.'+args.net[0])

    # Create the generator
    netG = arch.Generator(ngpu,nc, nz, True).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    # print(netG)

    # Create the Discriminator
    netD = arch.Discriminator(ngpu,nc, True).to(device)

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

    t = open(save_path+"text.txt", "a")
    print("Starting Training Loop...")
    t.close()
    # For each epoch
    for epoch in range(num_epochs):
        t = open(save_path+"text.txt", "a")
        t.write(str(epoch)+str(", "))
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
                np.save(save_path+"produced.npy",img_list)

            iters += 1

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(save_path+"loss.png")
        plt.close()

    PATH = "netG.pt"
    torch.save(netG, save_path+PATH)
    
    PATH = "netD.pt"
    torch.save(netD, save_path+PATH)
    
    # fig = plt.figure(figsize=(8,8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # HTML(ani.to_jshtml())
        


if __name__ == '__main__':
    print('starting')    
    run()
    sys.stdout.close()

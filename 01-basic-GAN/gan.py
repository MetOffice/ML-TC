import argparse
import os
import random
from datetime import datetime
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
import os, pathlib
import random
import sys
import net_architectures

# Define CUDA env variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description='Run GAN With Cyclone Data')
parser.add_argument('format', metavar='format', type=str, nargs=1, choices=['A','B','C','D','E'], help='data format to use')
parser.add_argument('net', metavar='net', type=str, nargs=1, choices=["Large_Net", "Small_Net"], help='which network variation to load')

parser.add_argument('path', metavar='path', default="/lustre/projects/metoffice/ml-tc/", type=str, nargs="?", help='the base path to use')
parser.add_argument('--vars', nargs="+", help="the set of variables to use")

args = parser.parse_args()
print(args)
# input()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Set Global Vars ----------------------------------------------
DATAPATH = args.path + '/Data/'  # Data directory
OUTPATH = args.path + '/output/01-basic-GAN/'  # Ouput directory

CHECK_CUDA = True  # Optional: print basic CUDA checks to output file
NGPU = torch.cuda.device_count()  # Number of GPUs available. Use 0 for CPU mode.

WORKERS = 0  # Number of workers for dataloader
BATCH_SIZE = 128  # Batch size during training
IMAGE_SIZE = 256  # Spatial size of training images. All images will be resized to this size using a transformer.
CHANNELS = 1  # Number of channels in the training images. For color images this is 3

NZ = 100  # Size of z latent vector (i.e. size of generator input)
NGF = 64  # Size of feature maps in generator
NDF = 64  # Size of feature maps in discriminator

NUM_EPOCHS = 1000  # Number of training epochs
LR = 0.0002  # Learning rate for optimizers

BETA1 = 0.5  # Beta1 hyperparam for Adam optimizers

PRINT_FREQ = 50  # Frequency of printing training stats (in steps)
TEST_FREQ = 500  # Frequency of testing the generator (in steps)

SEED = 357  # Set random seed for reproducibility
# SEED = random.randint(1, 10000) # use if you want new results

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def print_CUDA_status():
    """
    Print some basic CUDA checks
    """
    print('-'*20)
    print(f'CUDA is available: {str(torch.cuda.is_available())}')
    print(f'Visible GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    print(f'pytorch {torch.__version__}')
    print('-'*20)


def run():
    if CHECK_CUDA: print_CUDA_status()
    
    base_path=args.path

    # Create dedicated output directory for this trial
    save_path=OUTPATH+datetime.now().strftime("%Y%m%dT%H%M/")
    os.mkdir(save_path)
    print(f'Saving this run to {save_path}')
    
    print("Run starting...")
    print("Random Seed: "+str(SEED))

    random.seed(SEED)
    torch.manual_seed(SEED)
    
    data=[]
    for f in os.listdir(DATAPATH):
        # print(len(data))
        if f.endswith(".npz") and args.format[0] in f:
            print(f'Data: {f}')
            # Load pickled arrays
            datapoint=np.load(DATAPATH+f, allow_pickle=True)
            for x in datapoint['arr_0']:
                s=x.shape
                if s.count(s[0]) == len(s) and np.count_nonzero(~np.isnan(x)) > 3000:
                    # print(x.shape)
                    data.append(torch.from_numpy(np.nan_to_num(x)))
    # print(len(data))
    random.shuffle(data)

    data=torch.stack(data)
    
    # Transform data TODO:Check for better method
    data=data/(data.max()/2)-1
    dataset=TensorDataset(data)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=WORKERS)
                                            
    # # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")
    # cuda0 = torch.device("cuda:0")
    # cuda1 = torch.device("cuda:1")

    # Get the right architecture class
    arch=eval('net_architectures.'+args.net[0])

    # Create the generator
    netG = arch.Generator(NGPU,CHANNELS, NZ, True).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (NGPU > 1):
        netG = nn.DataParallel(netG, list(range(NGPU)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the generator model
    print(netG)

    # Create the Discriminator
    netD = arch.Discriminator(NGPU,CHANNELS, True).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (NGPU > 1):
        netD = nn.DataParallel(netD, list(range(NGPU)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the discriminator model
    print(netD)
     
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch}')
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
            noise = torch.randn(b_size, NZ, 1, 1, device=device)
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
            if iters % PRINT_FREQ == 0:
                print(f'[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f} / \tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % TEST_FREQ == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
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

    # Save pytorch objects
    torch.save(netG, save_path+"netG.pt")    
    torch.save(netD, save_path+"netD.pt")
    
    # fig = plt.figure(figsize=(8,8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # HTML(ani.to_jshtml())
        


if __name__ == '__main__':
    run()
    sys.stdout.close()

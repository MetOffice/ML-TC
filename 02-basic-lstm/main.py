# import libraries
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process
import numpy as np
import random

from utils.start_tensorboard import run_tensorboard
from models.seq2seq_ConvLSTM import EncoderDecoderConvLSTM
from data.MovingMNIST import MovingMNIST

import argparse

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#print(torch.cuda.device_count())

DATAPATH="/projects/metoffice/ml-tc/ML-TC/Data/"
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')

opt = parser.parse_args()
#from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
"""
data=[]
for f in os.listdir(DATAPATH):
    # print(len(data))
    if f.endswith(".npz") and "C" in f:
        print(f'Data: {f}')
        # Load pickled arrays
        datapoint=np.load(DATAPATH+f, allow_pickle=True)
        #print(len(list(chunks(datapoint['arr_0'],int(len(datapoint['arr_0'])/7)))[-1]))
        for x in list(chunks(datapoint['arr_0'],7)):
            print(len(x))
            if len(x)==7:
                flag=True
                for y in x:
                    s=y.shape
                    if not s.count(s[0]) == len(s):
                        flag=False
                if flag:
                    data.append(torch.from_numpy(np.nan_to_num(x)))
print(len(data))
# print(len(data))
random.shuffle(data)

data=torch.stack(data)

# Transform data TODO:Check for better method
data=data/(data.max()/2)-1
"""

##########################
######### MODEL ##########
##########################

class MovingMNISTLightning(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(MovingMNISTLightning, self).__init__()

        # default config
        self.path = os.getcwd() + '/data'
        self.model = model

        # logging config
        self.log_images = True

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = opt.batch_size
        self.n_steps_past = 5
        self.n_steps_ahead = 2  # 4

    def create_video(self, x, y_hat, y):
        # predictions with input for illustration purposes
        preds = torch.cat([x.cpu(), y_hat.unsqueeze(2).cpu()], dim=1)[0]
        '''print("x")
        print(x.min())
        print(x.max())
        print(x[0,0,0,0,0].type())
        print("y")
        print(y.min())
        print(y.max())
        print(y[0,0,0,0].type())
        print("y_hat")
        print(y_hat.min())
        print(y_hat.max())
        print(y_hat[0,0,0,0].type())
'''
        # entire input and ground truth
        y_plot = torch.cat([x.cpu(), y.unsqueeze(2).cpu()], dim=1)[0]

        # error (l2 norm) plot between pred and ground truth
        difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        zeros = torch.zeros(difference.shape)
        difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[
            0].unsqueeze(1)

        # concat all images
        final_image = torch.cat([preds, y_plot, difference_plot], dim=0)
        np.save('final_image.npy', final_image.cpu().detach().numpy())
        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=self.n_steps_past + self.n_steps_ahead)

        return grid

    def forward(self, x):
        x = x.to(device='cuda')

        output = self.model(x, future_seq=self.n_steps_ahead)

        return output

    def training_step(self, batch, batch_idx):
        #print(type(batch))
        #print(len(batch))
        #print(type(batch[0]))
        #print(len(batch[0]))
        batch=torch.unsqueeze(batch[0],2)
        #print(batch.min())
        #print(batch.max())
        #print(batch[0,0,0,0,0].type())
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        #x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()
        #print("size of x")
        #print(x.size())
        y_hat = self.forward(x).squeeze()  # is squeeze neccessary?

        loss = self.criterion(y_hat, y)

        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        # save predicted images every 250 global_step
        if self.log_images:
            if self.global_step % 250 == 0:
                final_image = self.create_video(x, y_hat, y)

                self.logger.experiment.add_image(
                    'epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
                plt.close()

        tensorboard_logs = {'train_mse_loss': loss,
                            'learning_rate': lr_saved}

        return {'loss': loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}


    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))

    @pl.data_loader
    def train_dataloader(self):
        global data
        training_dataset=TensorDataset(self.train_dataset)
        print("DATASET")
        print(len(training_dataset))
        # Create the dataloader
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=self.batch_size,
                                        shuffle=True, num_workers=16)

        """
        train_data = MovingMNIST(
            train=True,
            data_root=self.path,
            seq_len=self.n_steps_past + self.n_steps_ahead,
            image_size=64,
            deterministic=True,
            num_digits=2)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True)
"""
        return train_loader

    @pl.data_loader
    def test_dataloader(self):
        global data
        test_dataset=TensorDataset(self.test_dataset)

        # Create the dataloader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                        shuffle=True, num_workers=16)

        """
        test_data = MovingMNIST(
            train=False,
            data_root=self.path,
            seq_len=self.n_steps_past + self.n_steps_ahead,
            image_size=64,
            deterministic=True,
            num_digits=2)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=True)
"""
        return test_loader

    def prepare_data(self):
        data=[] 
        for f in os.listdir(DATAPATH):
            # print(len(data))
            if f.endswith(".npz") and "C" in f and "separate" in f:
                print(f'Data: {f}')
                # Load pickled arrays
                datapoint=np.load(DATAPATH+f, allow_pickle=True)
                #print(len(datapoint['arr_0']))
                #print(datapoint['arr_0'])
                for z in datapoint['arr_0']:
                    #print(len(list(chunks(z,int(len(z)/7)))[-1]))
                    for x in list(chunks(z,7)):
                        #print(len(x))
                        #print(len(x[0]))
                        #print(x[0].shape) 
                        #print(np.stack(x).shape)
                        x=np.stack(x)
                        if len(x)==7:
                            flag=True
                            for y in x:
                                s=y.shape
                                if not s.count(s[0]) == len(s):
                                    flag=False
                            if flag:
                                #data.append(torch.stack([torch.from_numpy(np.nan_to_num(item)).float() for item in x]))
                                data.append(x)
                                #data.append(torch.from_numpy(np.nan_to_num(x)))
        #print(len(data))
        #print(type(data))
        #print(len(data[0]))
        #print(data[0].dtype)
        #print(type(data[0]))
        #print(data[0][0].dtype)
        #lst = [torch.from_numpy(np.nan_to_num(item).astype("uint8")) for item in data]
        lst = [torch.from_numpy(np.nan_to_num(item).astype("float")).float() for item in data]
        #print(type(lst[0]))
        random.shuffle(lst)
        #print(len(lst))
        data=torch.stack(lst)
        np.save('/home/mo-txirouch/Video-Prediction-using-PyTorch/split_data.npy',data.numpy())
        #print(data.max())
        #print("saved")
        # Transform data TODO:Check for better method
        data=(data-data.min())/(data.max()-data.min())
        #print(data.max())
        #print(data.min())
        self.train_dataset = data[:int(0.8*len(data))]
        self.test_dataset = data[int(0.8*len(data)):]


def run_trainer():
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=1)

    model = MovingMNISTLightning(model=conv_lstm_model)

    trainer = Trainer(max_epochs=opt.epochs,
                      gpus=opt.n_gpus,
                      distributed_backend='dp',
                      early_stop_callback=False,
                      use_amp=opt.use_amp
                      )

    trainer.fit(model)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    p1 = Process(target=run_trainer)                    # start trainer
    p1.start()
    p2 = Process(target=run_tensorboard(new_run=True))  # start tensorboard
    p2.start()
    p1.join()
    p2.join()




#from https://towardsdatascience.com/video-prediction-using-convlstm-with-pytorch-lightning-27b195fd21a2
# import libraries
import os
import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.utils.tensorboard as tensorboard
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process
from lstm_rnn import EncoderDecoderConvLSTM
from mnist_data_loader import MovingMNIST

import argparse

# Define CUDA env variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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

##########################
######### MODEL ##########
##########################

class MovingMNISTLightning(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(MovingMNISTLightning, self).__init__()

        # default config
        self.path = os.getcwd() + '/data'
        self.normalize = False
        self.model = model
        print(model)
        # logging config
        self.log_images = True

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = opt.batch_size
        self.n_steps_past = 10
        self.n_steps_ahead = 10  # 4

    def create_video(self, x, y_hat, y):
        # predictions with input for illustration purposes
       # print("create_video")
       # print(x.shape)
       # print(y_hat.shape)
        preds = torch.cat([x.cpu(), y_hat.permute(0,2,1,3,4).cpu()], dim=1)[0]

        # entire input and ground truth
       # print(x.shape)
       # print(y.shape)
        y_plot = torch.cat([x.cpu(), y.unsqueeze(2).cpu()], dim=1)[0]
        
       # print(preds.shape)
       # print(y_plot.shape)

        # error (l2 norm) plot between pred and ground truth
        difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        zeros = torch.zeros(difference.shape)
        difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[
            0].unsqueeze(2)
        difference_plot=torch.cat((difference_plot[0],difference_plot[1]),0)
        #print(difference_plot.shape)
        # concat all images
        final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=self.n_steps_past + self.n_steps_ahead)

        return grid

    def forward(self, x):
        x = x.to(device='cuda')

        output = self.model(x, future_seq=self.n_steps_ahead)

        return output

    def training_step(self, batch, batch_idx):
       #print(self.n_steps_past)
       # print(type(self.n_steps_past))
       # print(len(batch))
       # print(type(batch))
       # print(len(batch[0]))
       # print(len(batch[1]))
        temp_1=torch.unsqueeze(batch[0],2)
        temp_2=torch.unsqueeze(batch[1],2)
        #batch=torch.stack((temp_1,temp_2))
        #batch=temp_1
        batch=torch.cat((temp_1,temp_2),1)
       # print("bottom type")
       # print(batch[0,0,0,0,0])
       # print(batch[0,0,0,0,0].dtype)
        batch=batch.to(torch.float)
       # print("bottom type")
       # print(batch[0,0,0,0,0])
       # print(batch[0,0,0,0,0].dtype)
       # print(batch.shape)
        #torch.unsqueeze(batch,0)
       # print(type(batch))
       # print(type(batch[0]))
       # print(batch[:,:,:10,:,:].shape)
       # input()
       # x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x = batch[:, 0:self.n_steps_past, :, :,:]
        y = batch[:,self.n_steps_past:, :, :, :]
        #x = x.permute(0,2, 1,3,4)
        y = y.squeeze()
        #print("how tf")
        #print(x.shape)
        y_hat = self.forward(x)#.squeeze()  # is squeeze neccessary?
        #print("preds")
        #print(y_hat.shape)
        #print("truth")
        #print(y.shape)
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

    
    def train_dataloader(self):
        
        train_data = MovingMNIST(
            train=True,
            root=self.path,
            download=True) 
        #print("Type of data")
        #print(type(train_data))
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True)
        #print(type(iter(train_data)[0]))
        #print(iter(train_data)[0])
        return train_loader

    
    def test_dataloader(self):
        test_data = MovingMNIST(
            train=False,
            data_root=self.path)
            #seq_len=self.n_steps_past + self.n_steps_ahead,
            #image_size=64,
            #deterministic=True,
            #num_digits=2)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=True)

        return test_loader



def run_trainer():
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=1)

    model = MovingMNISTLightning(model=conv_lstm_model)

    trainer = Trainer(max_epochs=opt.epochs,
                      gpus=opt.n_gpus)

    trainer.fit(model)


if __name__ == '__main__':
    p1 = Process(target=run_trainer)                    # start trainer
    p1.start()
    p2 = Process(target=tensorboard(new_run=True))  # start tensorboard
    p2.start()
    p1.join()
    p2.join()

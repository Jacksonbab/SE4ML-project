import numpy as np
import argparse
import pandas as pd
import matplotlib.image as mpimg
from torch.autograd import Variable

# import h5py
import os
# from PIL import Image
# import PIL
import matplotlib.pyplot as plt
# import csv
from dataset_unsplit import MultiDirectoryDataSequence
import time
import sys
from tqdm import tqdm
from model import DAVE2PytorchModel, DAVE2v1, DAVE2v2, DAVE2v3, Epoch

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='parent directory of training dataset')
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--robustification", type=bool, default=True)
    parser.add_argument("--noisevar", type=int, default=15)
    parser.add_argument("--log_interval", type=int, default=80)
    args = parser.parse_args()
    return args


def characterize_steering_distribution(y_steering, generator):
    turning = []; straight = []
    for i in y_steering:
        if abs(i) < 0.1:
            straight.append(abs(i))
        else:
            turning.append(abs(i))
    # turning = [i for i in y_steering if i > 0.1]
    # straight = [i for i in y_steering if i <= 0.1]
    try:
        print("Moments of abs. val'd turning steering distribution:", generator.get_distribution_moments(turning))
        print("Moments of abs. val'd straight steering distribution:", generator.get_distribution_moments(straight))
    except Exception as e:
        print(e)
        print("len(turning)", len(turning))
        print("len(straight)", len(straight))

def evaluate(model, testloader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, hashmap in enumerate(testloader, 0):
            x = hashmap['image'].float().to(device)
            y = hashmap['steering_angle'].float().to(device)
            outputs = model(x)
            test_loss += F.mse_loss(outputs, y, reduction='sum').item() # sum up batch loss
    test_loss /= len(testloader.dataset)
    print(f"Test set MSE loss: {test_loss}")
    return test_loss

def main():
    start_time = time.time()
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    input_shape = (672, 188)
    #model = DAVE2v3(input_shape=input_shape)
    model = DAVE2PytorchModel(input_shape=input_shape)
    args = parse_arguments()
    print(args)
    dataset = MultiDirectoryDataSequence(args.dataset, image_size=(model.input_shape[::-1]), transform=Compose([ToTensor()]),\
                                         robustification=args.robustification, noise_level=args.noisevar) #, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    # split into train and test sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())
    print("Total samples:", dataset.get_total_samples())
    def worker_init_fn(worker_id):
        np.random.seed(42)

    trainloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True, worker_init_fn=worker_init_fn)
    print("splitting dataset into train and test sets...")
    print("train size:", len(train_dataset), "test size:", len(test_dataset))
    
    print("time to load dataset: {}".format(time.time() - start_time))

    iteration = f'{model._get_name()}-{input_shape[0]}x{input_shape[1]}-lr{args.lr}-{args.epochs}epoch-{args.batch}batch-lossMSE-{int(dataset.get_total_samples()/1000)}Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"{device=}")
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.9, 0.999), eps=1e-08)
    lowest_loss = 1e5
    # logfreq = args.log_interval
    logfreq = len(trainloader)
    cur_step = 0
    train_losses = []
    test_losses = []
    
    for epoch in range(args.epochs):
        print(f"Starting {epoch=}...")
        running_loss = 0.0
        for i, hashmap in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            x = hashmap['image'].float().to(device)
            y = hashmap['steering_angle'].float().to(device)
            # x = Variable(x, requires_grad=True)
            # y = Variable(y, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            # loss = F.mse_loss(outputs.flatten(), y)
            loss = F.mse_loss(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if cur_step % logfreq == logfreq-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / logfreq))
                model.eval()
                test_loss = evaluate(model, testloader, device)
                test_losses.append(test_loss)
                if test_loss < lowest_loss:
                    print(f"New best model! MSE loss: {test_loss}")
                    model_name = f"D:/Academic/SE for ML/final_project/model_cpt/model-{time_stamp}-fixnoise-{iteration}-best.pt"
                    print(f"Saving model to {model_name}")
                    torch.save(model, model_name)
                    lowest_loss = test_loss
                model.train()
            cur_step += 1

        train_loss = running_loss / len(trainloader.dataset)
        train_losses.append(train_loss)
        print(f"Train set MSE loss: {train_loss}")
        running_loss = 0.0  
        print(f"Finished {epoch=}")
        model_name = f"D:/Academic/SE for ML/final_project/model_cpt/model-{time_stamp}-fixnoise-{iteration}-epoch{epoch}.pt"
        print(f"Saving model to {model_name}")
        torch.save(model, model_name)
        # if loss < 0.002:
        #     print(f"Loss at {loss}; quitting training...")
        #     break
    print('Finished Training')

    # save model
    # torch.save(model.state_dict(), f'H:/GitHub/DAVE2-Keras/test{iteration}-weights.pt')
    model_name = f'D:/Academic/SE for ML/final_project/model_cpt/model-{time_stamp}-{iteration}-last.pt'
    torch.save(model, model_name)

    # # delete models from previous epochs
    # print("Deleting models from previous epochs...")
    # for epoch in range(args.epochs):
    #     os.remove(f"D:/Academic/SE for ML/final_project/model_cpt/model-{iteration}-epoch{epoch}.pt")
    print(f"Saving model to {model_name}")
    print("All done :)")
    time_to_train=time.time() - start_time
    print("Time to train: {}".format(time_to_train))
    # save metainformation about training
    with open(f'D:/Academic/SE for ML/final_project/model_cpt/model-{time_stamp}-{iteration}-metainfo.txt', "w") as f:
        f.write(f"{model_name=}\n"
                f"total_samples={len(dataset)}\n"
                f"{args.epochs=}\n"
                f"{args.lr=}\n"
                f"{args.batch=}\n"
                f"{optimizer=}\n"
                f"final_loss={running_loss / logfreq}\n"
                f"{device=}\n"
                f"{args.robustification=}\n"
                f"{args.noisevar=}\n"
                f"dataset_moments={dataset.get_outputs_distribution()}\n"
                f"{time_to_train=}\n"
                f"dirs={dataset.get_directories()}")

    # save all losses as numpy arrays
    losses = {"train_losses": np.array(train_losses), "test_losses": np.array(test_losses)}
    np.savez(f'D:/Academic/SE for ML/final_project/model_cpt/model-{time_stamp}-{iteration}-losses.npz', **losses)

    # plot losses
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.savefig(f'D:/Academic/SE for ML/final_project/model_cpt/model-{time_stamp}-{iteration}-losses.png')



        



if __name__ == '__main__':
    main()
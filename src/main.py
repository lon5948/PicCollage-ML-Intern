import argparse
import torch

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import RegressionDataset
from train import train
from preprocess import preprocessing


def main(args):
    print(f'Preprocessing data...')
    x_train, y_train, x_test, y_test = preprocessing(args.dir)
    print("Loading data...")
    transform = transforms.Compose([
        transforms.ToPILImage(),                                    
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(RegressionDataset(x_train, y_train, transform), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(RegressionDataset(x_test, y_test, transform), batch_size=args.batch_size, shuffle=False)
    print ('Start training process...') 
    train(args.lr, args.epoch, train_loader, test_loader)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Correlation Prediction")
    parser.add_argument("--dir", type=str, default='./data', help="path of data directory (default: './data')")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)")
    parser.add_argument("--epoch", type=int, default=10, help="number of epochs (default: 10)")
    main(parser.parse_args())

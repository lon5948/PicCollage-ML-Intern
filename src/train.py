import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from model import Regression


def train(lr: float, epochs: int, train_loader: DataLoader, test_loader: DataLoader):
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model = Regression().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)    
    criterion = nn.MSELoss().to(device)
   
    for epoch in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        min_test_loss = 0.0
        
        print(f"Epoch {epoch+1}:")
        model.train()
        for (image, corr) in tqdm(train_loader):
            if epoch == 0: # check the Loss before training
                break
            image = image.to(device)
            target = corr.to(device).unsqueeze(1)
            train_pred = model(image)
            loss = criterion(train_pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            for (image, corr) in tqdm(test_loader):
                image = image.to(device)
                target = corr.to(device).unsqueeze(1)
                test_pred = model(image)
                loss = criterion(test_pred, target)
                test_loss += loss.item()
            
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                torch.save(model.state_dict(), "best_model.pkl")
            elif test_loss > min_test_loss:
                break
        
            print(f"Ground Truth: {corr[-1]}\nPrediction: {test_pred[-1].item()}")
            print(f"Train Loss: {train_loss/len(train_loader)} | Test Loss: {test_loss/len(test_loader)}\n---")

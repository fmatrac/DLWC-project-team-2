# training.py

import torch
import torch.nn as nn

def train(model, train_loader, epochs=50, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)   
            y = batch["y"].to(device)  

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:3d}  train_loss={avg_loss:.6f}")

def validate(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()

    total_mse  = 0.0
    total_mae  = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            x      = batch["x"].to(device)   
            target = batch["y"].to(device)   

            output = model(x)                

            total_mse     += criterion(output, target).item() * len(target)
            total_mae     += (output - target).abs().sum().item()
            total_samples += len(target)

    mse = total_mse / total_samples
    mae = total_mae / total_samples

    print(f"Test MSE : {mse:.6f}")
    print(f"Test MAE : {mae:.6f}  (avg absolute return error)")
    return mse, mae

        

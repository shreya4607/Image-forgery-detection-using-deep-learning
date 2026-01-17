num_epochs = 25
train_losses, val_losses = [], []

best_val_loss = float('inf')

BEST_MODEL_PATH = "best_casia_densenet.pth"


patience = 1
counter = 0




for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0

    for rgb_imgs, dct_feats, labels in train_loader:
        rgb_imgs = rgb_imgs.to(device)
        dct_feats = dct_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(rgb_imgs, dct_feats).view(-1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss) 

    model.eval()
    running_val_loss = 0
    with torch.no_grad():
        for rgb_imgs, dct_feats, labels in val_loader:
            rgb_imgs = rgb_imgs.to(device)
            dct_feats = dct_feats.to(device)
            labels = labels.to(device)

            outputs = model(rgb_imgs, dct_feats).view(-1)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)       # ✅ ADD THIS

    print(f"Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

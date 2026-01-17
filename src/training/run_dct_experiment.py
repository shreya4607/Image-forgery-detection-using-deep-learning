device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_DCT_Fusion().to(device)


train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = \
    load_casia_data(
        "/kaggle/input/c-casia-v2/C-CASIA2/Au/Au",
        "/kaggle/input/c-casia-v2/C-CASIA2/Tp/C_Tp"
    )


from torchvision import transforms

rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_ds = CASIADatasetDCT(train_imgs, train_lbls, rgb_transform)
val_ds   = CASIADatasetDCT(val_imgs, val_lbls, rgb_transform)
test_ds  = CASIADatasetDCT(test_imgs, test_lbls, rgb_transform)


train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


model = model.to(device)

# Freeze CNN backbone
for p in model.cnn.features.parameters():
    p.requires_grad = False

# Loss
criterion = nn.BCEWithLogitsLoss()

# Optimizer: train ONLY DCT MLP + fusion head
trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(
    trainable_params,
    lr=1e-4,          # 🔥 IMPORTANT CHANGE
    weight_decay=1e-4
)


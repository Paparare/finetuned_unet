from diffusers import UNet2DModel
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn, optim
from PIL import Image
import os, torch
from torchvision import transforms
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

class TextileDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_images = os.listdir(input_dir)
        self.output_images = os.listdir(output_dir)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_dir, self.input_images[idx])
        output_img_path = os.path.join(self.output_dir, self.output_images[idx])

        input_image = Image.open(input_img_path).convert("RGB")
        output_image = Image.open(output_img_path).convert("L")

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image


# Load UNet model
def create_model():
    return UNet2DModel(
    sample_size=256,
    in_channels=3,
    out_channels=1,
    layers_per_block=1,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
).cuda()


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = TextileDataset(input_dir='pic/input', output_dir='pic/output', transform=transform)

kf = KFold(n_splits=5, shuffle=True)

num_epochs = 10
batch_size = 2
fold = 1

for train_index, val_index in kf.split(dataset):
    print(f'Fold {fold}:')

    train_subset = Subset(dataset, train_index)
    val_subset = Subset(dataset, val_index)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = create_model()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)

            optimizer.zero_grad()

            with autocast():
                timestep = torch.tensor([1.0], device=model.device)
                outputs = model(inputs, timestep)
                predictions = outputs.sample

                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader)}')

    model.save_pretrained(f'finetuned_model_fold_{fold}', safe_serialization=False)
    fold += 1

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
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
).cuda()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = TextileDataset(input_dir='pic/input', output_dir='pic/output', transform=transform)

kf = KFold(n_splits=5, shuffle=True)

num_epochs = 100
batch_size = 4
fold = 1
early_stop_patience = 5  # Number of epochs to wait for improvement

for train_index, val_index in kf.split(dataset):
    print(f'Fold {fold}:')

    # Split data into training and validation sets
    train_subset = Subset(dataset, train_index)
    val_subset = Subset(dataset, val_index)

    # Create dataloaders for training and validation
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Initialize the model for each fold
    model = create_model()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Reset early stopping variables for each fold
    best_val_loss = float('inf')  # Initialize to a very large number
    no_improvement_epochs = 0  # Counter to track epochs with no improvement

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)

            optimizer.zero_grad()
            timestep = torch.tensor([1.0], device=model.device)
            outputs = model(inputs, timestep)
            predictions = outputs.sample

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(model.device), val_targets.to(model.device)
                val_outputs = model(val_inputs, timestep)
                val_predictions = val_outputs.sample

                val_loss += criterion(val_predictions, val_targets).item()

        val_loss /= len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0  # Reset counter when validation loss improves
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break

    # Save the model for the current fold
    print("model saved!!!!!!!!!!!!!!!")
    model.save_pretrained(f'resized_finetuned_model_fold_{fold}')
    fold += 1

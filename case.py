from diffusers import UNet2DModel
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Paths
input_image_path = 'case1.png'

# Convert image to tensor without resizing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Convert image to tensor without resizing
])

# Open the image and apply transformations
input_image = Image.open(input_image_path).convert('RGB')  # Ensure it's RGB
input_tensor = preprocess(input_image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

# Function to pad image to nearest multiple of 32 (or 64)
def pad_to_multiple_of(image_tensor, multiple=32):
    _, _, h, w = image_tensor.size()  # Get height and width of the image
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')

# Pad the image to ensure compatible dimensions
input_tensor = pad_to_multiple_of(input_tensor, multiple=32)

# Loop through each fold and process the image using each fine-tuned model
for fold in range(1, 6):  # Assuming 5 folds
    # Load the fine-tuned model for the current fold with safe_serialization set to False
    model_path = f'resized_finetuned_model_fold_{fold}'
    model = UNet2DModel.from_pretrained(model_path)
    model = model.cuda()  # Move to GPU if available
    model.eval()  # Set to evaluation mode

    # Define a timestep (if applicable in your case, you can use 1.0 for standard processing)
    timestep = torch.tensor([1.0], device=input_tensor.device)

    # Run inference (forward pass)
    with torch.no_grad():
        output = model(input_tensor, timestep)
        processed_tensor = output.sample  # Get the prediction from the output

    # Convert the output tensor back to an image
    processed_image = processed_tensor.squeeze(0).cpu()  # Remove batch dimension and move to CPU
    processed_image = transforms.ToPILImage()(processed_image)  # Convert tensor to PIL Image

    # Save the processed image with the fold number in the file name
    output_image_path = f'resized_case1_processed_fold_{fold}.png'
    processed_image.save(output_image_path)

    print(f"Processed image saved at {output_image_path}")

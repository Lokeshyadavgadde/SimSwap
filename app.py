import gradio as gr
import torch
from PIL import Image
from models.models import create_model
from options.test_options import TestOptions
import torchvision.transforms as transforms

# Initialize the model
opt = TestOptions().parse()
model = create_model(opt)
model.setup(opt)
model.eval()

# Define the face swapping function
def swap_faces(source_image, target_image):
    # Preprocess images
    transform = transforms.Compose([
        transforms.Resize((opt.crop_size, opt.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    source_tensor = transform(source_image).unsqueeze(0)
    target_tensor = transform(target_image).unsqueeze(0)

    # Perform face swapping
    with torch.no_grad():
        result = model(source_tensor, target_tensor)
    result_image = transforms.ToPILImage()(result.squeeze(0).cpu())
    return result_image

# Create Gradio interface
iface = gr.Interface(
    fn=swap_faces,
    inputs=[gr.Image(type="pil"), gr.Image(type="pil")],
    outputs=gr.Image(type="pil"),
    title="SimSwap Face Swapper",
    description="Upload a source and target image to perform face swapping."
)

if __name__ == "__main__":
    iface.launch()

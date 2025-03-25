from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageFilter
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models

app = FastAPI()

# Load the base model
model = models.efficientnet_b0(pretrained=False)

# Modify the classifier to match the checkpoint's output size
num_classes = 2  # Adjust this to match the saved model
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)

# Load the state_dict
model.load_state_dict(torch.load("notebook/efficientnet_b0_v2.pth", map_location=torch.device("cpu")))

# Set to evaluation mode
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/blur-image")
async def blur_image(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Apply transformations and model prediction
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Blur if classified as category 1
        if predicted_class == 1:
            image = image.filter(ImageFilter.GaussianBlur(radius=50))

        # Save and return the image
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

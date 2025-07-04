import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore warnings from torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont

# === Settings ===
img_path   = r'C:\Users\User\Downloads\images.jpg'  # your input
output_path = r'C:\Users\User\Downloads\detections_torchvision.jpg'
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
conf_thresh = 0.5  # only keep detections > 50% confidence

# === Load model ===
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# === Prepare image ===
pil_img = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.ToTensor(),           # scales to [0,1] and converts to C×H×W
])
img_tensor = transform(pil_img).to(device)

# === Inference ===
with torch.no_grad():
    outputs = model([img_tensor])[0]

# === Filter for “bottle” (COCO class 44) ===
boxes      = outputs['boxes']
labels     = outputs['labels']
scores     = outputs['scores']

bottle_inds = (labels == 44) & (scores >= conf_thresh)
bottle_boxes  = boxes[bottle_inds].cpu().numpy()
bottle_scores = scores[bottle_inds].cpu().numpy()

print(f"Total bottles detected: {len(bottle_boxes)}")

# === Draw & save ===
draw = ImageDraw.Draw(pil_img)
font = ImageFont.load_default()
for (x1, y1, x2, y2), score in zip(bottle_boxes, bottle_scores):
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
    draw.text((x1, y1-10), f"bottle {score:.2f}", fill="lime", font=font)

pil_img.save(output_path)
print(f"Annotated image saved to {output_path}")
pil_img.show()

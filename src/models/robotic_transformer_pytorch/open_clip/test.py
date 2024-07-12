import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='/data2/liangxiwen/RM-PRT/IL/worldmodel/src/models/robotic_transformer_pytorch/clip_laion2b_s34b_b79k/open_clip_pytorch_model.bin')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print('_',_)
image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
print(image.device)
text = tokenizer(["a diagram", "a dog", "a cat"])
print(text.device)
# Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features, _ = model.encode_image(image)
    print('features',_.shape)
    print('image_features',image_features.shape)
    text_features = model.encode_text(text)
    print('text_features',text_features.shape)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
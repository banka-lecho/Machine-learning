import warnings

warnings.filterwarnings('ignore')
import torch
from PIL import Image
import os
from tqdm import tqdm
from open_clip import create_model_from_pretrained


def get_vit_embs(path_png):
    device = "cuda:1"
    dict_all_embs = {}
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP')
    # model.to(device)

    for filename in tqdm(os.listdir(path_png), desc="Берем ембеддинги"):
        image_path = os.path.join(path_png, filename)
        image = Image.open(image_path).convert('RGB')
        inputs = preprocess(image).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            # features = model.encode_image(inputs.to(device))
            features = model.encode_image(inputs)
        embeds = features.detach().cpu().numpy()
        dict_all_embs[image_path] = embeds

    return dict_all_embs

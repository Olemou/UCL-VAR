# eval.py
import torch
from model.vit import VisionTransformer

def load_model(checkpoint_path="./checkpoints/best_model.pth"):
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(variant=checkpoint.get("vit_variant", "base"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

if __name__ == "__main__":
    model = load_model()
    print("Model loaded and ready!")

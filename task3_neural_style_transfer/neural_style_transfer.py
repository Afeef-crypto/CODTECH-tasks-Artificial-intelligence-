import argparse
from typing import Dict, List

import torch
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms


def image_loader(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image


def save_image(tensor: torch.Tensor, output_path: str) -> None:
    image = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    to_pil = transforms.ToPILImage()
    to_pil(image).save(output_path)


def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    _, channels, height, width = feature_map.size()
    features = feature_map.view(channels, height * width)
    gram = torch.mm(features, features.t())
    return gram / (channels * height * width)


def get_features(x: torch.Tensor, model: torch.nn.Module, layers: Dict[str, str]) -> Dict[str, torch.Tensor]:
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def run_style_transfer(
    content_path: str,
    style_path: str,
    output_path: str,
    steps: int = 300,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 512 if torch.cuda.is_available() else 256

    content = image_loader(content_path, image_size, device)
    style = image_loader(style_path, image_size, device)
    target = content.clone().requires_grad_(True).to(device)

    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    layers = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1", "28": "conv5_1"}
    content_layer = "conv4_1"
    style_layers: List[str] = list(layers.values())

    content_features = get_features(content, vgg, layers)
    style_features = get_features(style, vgg, layers)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

    optimizer = optim.Adam([target], lr=0.01)

    for step in range(1, steps + 1):
        target_features = get_features(target, vgg, layers)

        content_loss = torch.mean((target_features[content_layer] - content_features[content_layer]) ** 2)
        style_loss = 0.0

        for layer in style_layers:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            layer_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_loss

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(
                f"Step {step}/{steps} | "
                f"Content Loss: {content_loss.item():.4f} | "
                f"Style Loss: {style_loss.item():.4f}"
            )

    save_image(target, output_path)
    print(f"Styled image saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--output", type=str, default="styled_output.jpg", help="Output image path")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps")
    args = parser.parse_args()

    run_style_transfer(args.content, args.style, args.output, steps=args.steps)


if __name__ == "__main__":
    main()

from PIL import Image
from torchvision import transforms


def build_image_transform(height: int = 64, max_width: int = 384):
    """
    Для MPS важно ограничивать ширину -> иначе S=H'*W' растёт и attention ест память.
    """
    def resize_keep_aspect_and_clip(img: Image.Image) -> Image.Image:
        w, h = img.size
        new_h = height
        new_w = int(w * (new_h / h))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        if img.size[0] > max_width:
            img = img.crop((0, 0, max_width, new_h))
        return img

    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("L")),
        transforms.Lambda(resize_keep_aspect_and_clip),
        transforms.ToTensor(),
    ])

from PIL import Image
from torchvision import transforms


def build_image_transform(height: int = 64, max_width: int = 384):
    """
    Resize with aspect ratio, no crop.
    Output tensor always has fixed height=height.
    Width is variable up to max_width.
    """

    def resize_keep_aspect(img: Image.Image) -> Image.Image:
        img = img.convert("L")

        w, h = img.size

        scale = height / h
        new_h = height
        new_w = int(w * scale)

        if new_w > max_width:
            scale = max_width / new_w
            new_w = max_width
            new_h = max(1, int(new_h * scale))

        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("L", (new_w, height), color=255)

        top = (height - new_h) // 2
        canvas.paste(img, (0, top))

        return canvas

    return transforms.Compose([
        transforms.Lambda(resize_keep_aspect),
        transforms.ToTensor(),
    ])
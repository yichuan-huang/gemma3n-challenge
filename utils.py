import base64
from PIL import Image
import os
import tempfile


def encode_image_to_base64(image_path):
    """
    Encodes an image file to a base64 string.
    Args:
        image_path (str): The path to the image file.
    Returns:
        str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_uploaded_file(uploaded_file):
    """
    Saves an uploaded file to a temporary location.
    """
    if not os.path.exists("temp"):
        os.makedirs("temp")

    temp_path = os.path.join("temp", uploaded_file.filename)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return temp_path


def resize_image(image_path, max_size=(1024, 1024)):
    """
    Resizes an image to fit within the specified maximum size while maintaining aspect ratio.
    Args:
        image_path (str): The path to the image file.
        max_size (tuple): The maximum width and height (width, height) for the resized image.
    """
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(image_path)

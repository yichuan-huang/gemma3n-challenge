from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import logging
from typing import Union, Tuple
from config import Config
from knowledge_base import GarbageClassificationKnowledge


class GarbageClassifier:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.knowledge = GarbageClassificationKnowledge()
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load the model and processor"""
        try:
            self.logger.info(f"Loading model: {self.config.MODEL_NAME}")

            # Load processor
            kwargs = {}
            if self.config.HF_TOKEN:
                kwargs["token"] = self.config.HF_TOKEN

            self.processor = AutoProcessor.from_pretrained(
                self.config.MODEL_NAME, **kwargs
            )

            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.MODEL_NAME,
                torch_dtype=self.config.TORCH_DTYPE,
                device_map=self.config.DEVICE_MAP,
            )

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to meet Gemma3n requirements (512x512)
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to 512x512 as required by Gemma3n
        target_size = (512, 512)

        # Calculate aspect ratio preserving resize
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:
            # Width is larger
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            # Height is larger or equal
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)

        # Resize image maintaining aspect ratio
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with target size and paste the resized image
        processed_image = Image.new(
            "RGB", target_size, (255, 255, 255)
        )  # White background

        # Calculate position to center the image
        x_offset = (target_size[0] - new_width) // 2
        y_offset = (target_size[1] - new_height) // 2

        processed_image.paste(image, (x_offset, y_offset))

        return processed_image

    def classify_image(self, image: Union[str, Image.Image]) -> Tuple[str, str]:
        """
        Classify garbage in the image

        Args:
            image: PIL Image or path to image file

        Returns:
            Tuple of (classification_result, full_response)
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Load and process image
            if isinstance(image, str):
                image = Image.open(image)
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be a PIL Image or file path")

            # Preprocess image to meet Gemma3n requirements
            processed_image = self.preprocess_image(image)

            # Prepare messages with system prompt and user query
            messages = [
                {
                    "role": "system",
                    "content": [{
                        "type" : "text",
                        "text": self.knowledge.get_system_prompt(),
                    }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_image},
                        {
                            "type": "text",
                            "text": "Please classify the garbage in this image and explain your reasoning.",
                        },
                    ],
                },
            ]

            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)
            input_len = inputs["input_ids"].shape[-1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                disable_compile=True,
            )
            response = self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
            )

            # Extract classification from response
            classification = self._extract_classification(response)

            # Create formatted response
            formatted_response = self._format_response(classification, response)

            return classification, formatted_response

        except Exception as e:
            self.logger.error(f"Error during classification: {str(e)}")
            import traceback

            traceback.print_exc()
            return "Error", f"Classification failed: {str(e)}"

    def _extract_classification(self, response: str) -> str:
        """Extract the main classification from the response"""
        categories = self.knowledge.get_categories()

        # Convert response to lowercase for matching
        response_lower = response.lower()

        # Look for exact category matches first
        for category in categories:
            if category.lower() in response_lower:
                return category

        # Look for key terms if no exact match
        category_keywords = {
            "Recyclable Waste": [
                "recyclable",
                "recycle",
                "plastic",
                "paper",
                "metal",
                "glass",
                "bottle",
                "can",
                "aluminum",
                "cardboard",
            ],
            "Food/Kitchen Waste": [
                "food",
                "kitchen",
                "organic",
                "fruit",
                "vegetable",
                "leftovers",
                "scraps",
                "peel",
                "core",
                "bone",
            ],
            "Hazardous Waste": [
                "hazardous",
                "dangerous",
                "toxic",
                "battery",
                "chemical",
                "medicine",
                "paint",
                "pharmaceutical",
            ],
            "Other Waste": [
                "other",
                "general",
                "trash",
                "garbage",
                "waste",
                "cigarette",
                "ceramic",
                "dust",
            ],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                return category

        return "Unable to classify"

    def _format_response(self, classification: str, full_response: str) -> str:
        """Format the response with classification and reasoning"""
        if not full_response.strip():
            return f"**Classification**: {classification}\n**Reasoning**: No detailed analysis available."

        # If response already contains structured format, return as is
        if "**Classification**" in full_response and "**Reasoning**" in full_response:
            return full_response

        # Otherwise, format it
        return f"**Classification**: {classification}\n\n**Reasoning**: {full_response}"

    def get_categories_info(self):
        """Get information about all categories"""
        return self.knowledge.get_category_descriptions()

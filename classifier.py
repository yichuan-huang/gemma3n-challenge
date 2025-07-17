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
            Tuple of (classification_result, detailed_analysis)
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
                    "content": [
                        {
                            "type": "text",
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
                            "text": "Please classify what you see in this image. If it shows garbage/waste items, classify them according to the garbage classification standards. If it shows people, living things, or other non-waste items, classify it as 'Unable to classify' and explain why it's not garbage.",
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
            )[0]

            # Extract classification from response
            classification = self._extract_classification(response)

            # Extract reasoning from response
            reasoning = self._extract_reasoning(response)

            return classification, reasoning

        except Exception as e:
            self.logger.error(f"Error during classification: {str(e)}")
            import traceback

            traceback.print_exc()
            return "Error", f"Classification failed: {str(e)}"

    def _extract_classification(self, response: str) -> str:
        """Extract the main classification from the response"""
        response_lower = response.lower()

        # First check for explicit "Unable to classify" statements
        unable_phrases = [
            "unable to classify",
            "cannot classify",
            "cannot be classified",
        ]

        if any(phrase in response_lower for phrase in unable_phrases):
            return "Unable to classify"

        # Check for non-garbage items (people, living things, etc.)
        non_garbage_indicators = [
            "person",
            "people",
            "human",
            "face",
            "man",
            "woman",
            "boy",
            "girl",
            "living",
            "alive",
            "animal",
            "pet",
            "dog",
            "cat",
            "bird",
            "elon musk",
            "celebrity",
            "famous person",
            "portrait",
            "photo of a person",
            "human being",
        ]

        # Check for explicit statements about not being garbage/waste
        non_waste_phrases = [
            "not garbage",
            "not waste",
            "not trash",
            "this is not",
            "does not appear to be waste",
            "not intended to be discarded",
            "not something that should be",
            "appears to be a person",
            "shows a person",
            "image of a person",
        ]

        # Only classify as "Unable to classify" if it's clearly not garbage
        if any(indicator in response_lower for indicator in non_garbage_indicators):
            return "Unable to classify"

        if any(phrase in response_lower for phrase in non_waste_phrases):
            return "Unable to classify"

        # Now look for waste categories - check exact matches first
        categories = self.knowledge.get_categories()
        waste_categories = [cat for cat in categories if cat != "Unable to classify"]

        for category in waste_categories:
            if category.lower() in response_lower:
                return category

        # Look for category keywords
        category_keywords = {
            "Recyclable Waste": [
                "recyclable",
                "recycle",
                "plastic",
                "paper",
                "metal",
                "glass",
                "aluminum",
                "foil",
                "can",
                "bottle",
                "cardboard",
                "tin",
                "steel",
                "iron",
                "copper",
                "brass",
                "recyclable material",
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
                "food waste",
                "organic waste",
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
                "hazardous waste",
            ],
            "Other Waste": [
                "cigarette",
                "ceramic",
                "dust",
                "diaper",
                "tissue",
                "general waste",
                "other waste",
            ],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                return category

        # If no clear classification found, default to "Unable to classify"
        return "Unable to classify"

    def _extract_reasoning(self, response: str) -> str:
        """Extract only the reasoning content, removing all formatting markers and classification info"""
        import re

        # Remove all formatting markers
        cleaned_response = response.replace("**Classification**:", "")
        cleaned_response = cleaned_response.replace("**Reasoning**:", "")
        cleaned_response = re.sub(
            r"\*\*.*?\*\*:", "", cleaned_response
        )  # Remove any **text**: patterns
        cleaned_response = cleaned_response.replace(
            "**", ""
        )  # Remove remaining ** markers

        # Remove category names that might appear at the beginning
        categories = self.knowledge.get_categories()
        for category in categories:
            if cleaned_response.strip().startswith(category):
                cleaned_response = cleaned_response.replace(category, "", 1)
                break

        # Split into sentences and clean up
        sentences = []

        # Split by common sentence endings
        parts = re.split(r"[.!?]\s+", cleaned_response)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Skip parts that are just category names
            if part in categories:
                continue

            # Skip parts that start with category names
            is_category_line = False
            for category in categories:
                if part.startswith(category):
                    is_category_line = True
                    break

            if is_category_line:
                continue

            # Clean up the sentence
            part = re.sub(
                r"^[A-Za-z\s]+:", "", part
            ).strip()  # Remove "Category:" type prefixes

            if part and len(part) > 3:  # Only keep meaningful content
                sentences.append(part)

        # Join sentences and ensure proper punctuation
        reasoning = ". ".join(sentences)
        if reasoning and not reasoning.endswith((".", "!", "?")):
            reasoning += "."

        return reasoning if reasoning else "Analysis not available"

    def get_categories_info(self):
        """Get information about all categories"""
        return self.knowledge.get_category_descriptions()

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
        """Extract the main classification from the response with enhanced logic"""
        response_lower = response.lower()

        # Strong indicators that this is NOT garbage - check these first
        non_garbage_indicators = [
            "unable to classify",
            "cannot classify",
            "not garbage",
            "not waste",
            "not trash",
            "person",
            "people",
            "human",
            "face",
            "man",
            "woman",
            "living",
            "alive",
            "animal",
            "pet",
            "dog",
            "cat",
            "functioning",
            "in use",
            "working",
            "operational",
            "furniture",
            "appliance",
            "electronic device",
            "building",
            "house",
            "room",
            "landscape",
            "vehicle",
            "car",
            "truck",
            "bike",
            "elon musk",
            "celebrity",
            "famous person",
            "portrait",
            "photo of a person",
        ]

        # Check for explicit statements about not being garbage
        non_garbage_phrases = [
            "this is not",
            "this does not appear to be",
            "not intended to be discarded",
            "not something that should be",
            "appears to be a person",
            "shows a person",
            "image of a person",
            "human being",
            "living creature",
        ]

        # First priority: Check for strong non-garbage indicators
        if any(indicator in response_lower for indicator in non_garbage_indicators):
            return "Unable to classify"

        # Second priority: Check for phrases indicating it's not garbage
        if any(phrase in response_lower for phrase in non_garbage_phrases):
            return "Unable to classify"

        # Third priority: Look for reasoning that explicitly says it's not waste/garbage
        reasoning_against_waste = [
            "cannot be classified as waste",
            "should not be classified as",
            "not appropriate to classify",
            "does not belong to any waste category",
            "is not waste material",
        ]

        if any(phrase in response_lower for phrase in reasoning_against_waste):
            return "Unable to classify"

        # Only if none of the above conditions are met, then look for garbage categories
        categories = self.knowledge.get_categories()
        waste_categories = [cat for cat in categories if cat != "Unable to classify"]

        # Look for exact category matches
        for category in waste_categories:
            if category.lower() in response_lower:
                # Double check - make sure the context is positive
                category_index = response_lower.find(category.lower())
                context_before = response_lower[
                    max(0, category_index - 50) : category_index
                ]
                context_after = response_lower[category_index : category_index + 50]

                # If there are negation words around the category, skip it
                negation_words = [
                    "not",
                    "cannot",
                    "unable",
                    "doesn't",
                    "isn't",
                    "won't",
                    "shouldn't",
                ]
                if any(
                    neg in context_before or neg in context_after
                    for neg in negation_words
                ):
                    continue

                return category

        # Look for key terms only if no explicit non-garbage indicators were found
        category_keywords = {
            "Recyclable Waste": [
                "recyclable",
                "recycle",
                "plastic bottle",
                "aluminum can",
                "cardboard box",
                "glass bottle",
                "metal can",
            ],
            "Food/Kitchen Waste": [
                "food scraps",
                "fruit peel",
                "vegetable waste",
                "leftovers",
                "organic waste",
                "kitchen waste",
            ],
            "Hazardous Waste": [
                "battery",
                "chemical container",
                "medicine bottle",
                "paint can",
                "toxic material",
            ],
            "Other Waste": ["cigarette butt", "ceramic piece", "dust", "general waste"],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                return category

        # Default to "Unable to classify" if nothing clear is found
        return "Unable to classify"

    def _extract_reasoning(self, response: str) -> str:
        """Extract just the reasoning part from the response"""
        # Remove formatting markers
        cleaned_response = response.replace("**Classification**:", "")
        cleaned_response = cleaned_response.replace("**Reasoning**:", "")
        cleaned_response = cleaned_response.replace("**", "")

        # Split by common separators and try to find the reasoning part
        lines = cleaned_response.split("\n")
        reasoning_lines = []

        skip_keywords = ["classification", "category"]

        for line in lines:
            line = line.strip()
            if line and not any(keyword in line.lower() for keyword in skip_keywords):
                reasoning_lines.append(line)

        if reasoning_lines:
            return " ".join(reasoning_lines)

        # Fallback: return the whole response cleaned up
        return cleaned_response.strip()

    def get_categories_info(self):
        """Get information about all categories"""
        return self.knowledge.get_category_descriptions()

# Check if running in Hugging Face Spaces environment
try:
    import spaces

    HF_SPACES = True
    print("Running in Hugging Face Spaces environment")
except ImportError:
    HF_SPACES = False
    print("Running in local environment")

import gradio as gr
from PIL import Image
import os
from classifier import GarbageClassifier
from config import Config

# Initialize classifier
config = Config()
classifier = GarbageClassifier(config)

# Load model at startup
print("Loading model...")
classifier.load_model()
print("Model loaded successfully!")


def classify_garbage_impl(image):
    """
    Actual classification implementation
    """
    if image is None:
        return "Please upload an image", "No image provided"

    try:
        classification, full_response, confidence_score = classifier.classify_image(image)
        confidence_display = f"{confidence_score}/10"
        return classification, full_response, confidence_display
    except Exception as e:
        return "Error", f"Classification failed: {str(e)}", "0/10"

# Apply GPU decorator based on environment
if HF_SPACES:
    classify_garbage = spaces.GPU(classify_garbage_impl)
    print("GPU decorator applied for Hugging Face Spaces")
else:
    classify_garbage = classify_garbage_impl
    print("Running without GPU decorator")


def get_example_images():
    """Get example images if they exist"""
    example_dir = "test_images"
    examples = []
    if os.path.exists(example_dir):
        for file in os.listdir(example_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                examples.append(os.path.join(example_dir, file))
    return examples[:10]  # Limit to 3 examples


# Create Gradio interface
with gr.Blocks(title="Garbage Classification System") as demo:
    gr.Markdown("# 🗂️ Garbage Classification System")
    gr.Markdown(
        "Upload an image to classify garbage into: Recyclable Waste, Food/Kitchen Waste, Hazardous Waste, or Other Waste"
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Garbage Image")

            classify_btn = gr.Button("Classify Garbage", variant="primary", size="lg")

        with gr.Column():
            classification_output = gr.Textbox(
                label="Classification Result",
                placeholder="Upload an image and click classify",
            )

            confidence_output = gr.Textbox(
                label="Confidence Score",
                placeholder="Confidence score will appear here",
            )

            full_response_output = gr.Textbox(
                label="Detailed Analysis",
                placeholder="Detailed reasoning will appear here",
                lines=10,
            )

    # Category information
    with gr.Accordion("📋 Garbage Categories Information", open=False):
        try:
            category_info = classifier.get_categories_info()
            for category, description in category_info.items():
                gr.Markdown(f"**{category}**: {description}")
        except Exception as e:
            gr.Markdown(f"Categories information not available: {str(e)}")

    # Examples section
    examples = get_example_images()
    if examples:
        gr.Examples(examples=examples, inputs=image_input, label="Example Images")

    # Event handlers
    classify_btn.click(
        fn=classify_garbage,
        inputs=image_input,
        outputs=[classification_output, full_response_output, confidence_output]
    )

    # Auto-classify on image upload
    image_input.change(
        fn=classify_garbage,
        inputs=image_input,
        outputs=[classification_output, full_response_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch()

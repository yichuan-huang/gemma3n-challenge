# Waste Classification Application

This project is a web-based application that classifies waste materials from user-uploaded images. It identifies the type of waste (e.g., cardboard, glass, metal) and provides information on how to properly dispose of it.

## Features

*   **Image-based classification:** Upload an image of a waste item to have it automatically classified.
*   **Multiple waste categories:** The application can identify a variety of waste materials.
*   **Disposal information:** After classification, the app provides guidance on how to dispose of the identified waste material.
*   **Web interface:** A user-friendly web interface built with Gradio makes the application easy to use.

## How it works

The application uses a pre-trained Vision Transformer (ViT) model to perform the image classification. The model has been fine-tuned on a dataset of waste images to accurately identify different materials. The disposal information is retrieved from a knowledge base within the application.

## Getting Started

### Prerequisites

*   Python 3.7+
*   Pip

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yichuan-huang/gemma3n-challenge
    ```
2.  Navigate to the project directory:
    ```bash
    cd gemma3n-challenge
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the application

To start the application, run the following command:

```bash
python app.py
```

This will launch a Gradio web server. You can access the application by opening the provided URL in your web browser.

## Project Structure

*   `app.py`: The main application file, containing the Gradio interface and the classification logic.
*   `classifier.py`:  Handles the image classification using the pre-trained model.
*   `config.py`: Contains configuration settings for the application, such as the model name and labels.
*   `knowledge_base.py`:  A simple knowledge base containing disposal information for different waste materials.
*   `requirements.txt`: A list of the Python dependencies required to run the application.
*   `test_images/`: A directory containing sample images for testing the application.

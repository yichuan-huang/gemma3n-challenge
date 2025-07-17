# Waste Classification Application

This project is a web-based application that classifies waste materials from user-uploaded images. It identifies the type of waste (e.g., cardboard, glass, metal) and provides information on how to properly dispose of it.

## 🚀 Live Demo

Try the application live on Hugging Face Spaces!

**➡️ [Waste Classification Demo](https://huggingface.co/spaces/HMWCS/Gemma3n-challenge-demo)**

---

## ✨ Features

* **Image-based classification:** Upload an image of a waste item to have it automatically classified.
* **Multiple waste categories:** The application can identify a variety of waste materials.
* **Disposal information:** After classification, the app provides guidance on how to dispose of the identified waste material.
* **Web interface:** A user-friendly web interface built with Gradio makes the application easy to use.

---

## 💡 How it works

The application uses a pre-trained Gemma3n(E2B) model to perform the image classification. The model has been fine-tuned on a dataset of waste images to accurately identify different materials. The disposal information is retrieved from a knowledge base within the application.

---

## 🛠️ Getting Started

### Prerequisites

* Python 3.9+
* Pip
* Cuda (optional)

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yichuan-huang/gemma3n-challenge](https://github.com/yichuan-huang/gemma3n-challenge)
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
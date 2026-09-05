import random
from typing import cast
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from CNN.train import ResNet

CLASSES = (
    "Aereo",
    "Automobile",
    "Uccello",
    "Gatto",
    "Cervo",
    "Cane",
    "Rana",
    "Cavallo",
    "Nave",
    "Camion",
)

#Caricamento Modello
model = ResNet()
model.load_state_dict(
    torch.load("C:\\Users\\Giovanni\\ai\\CNN\\resnet_cifar10.pth", map_location=torch.device("cpu"))
)
model.eval()

#Dataset
testset_raw = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True
)

transform_pipeline = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def predict(image):
    if image is None:
        return None

    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        pil_img = Image.fromarray(np.uint8(image))

    if pil_img.size != (32, 32):
        pil_img = pil_img.resize((32, 32))
    img_tensor = cast(torch.Tensor, transform_pipeline(pil_img))
    img_batch = img_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_batch)
        probabilities = F.softmax(outputs, dim=1)[0]

    return {CLASSES[i]: float(probabilities[i]) for i in range(10)}

#random image from CIFAR-10 dataset
def get_random_cifar_image():
    idx = random.randint(0, len(testset_raw) - 1)
    img, label = testset_raw[idx]

    img_np = np.array(img)
    true_label_text = f"Classe Reale: {CLASSES[label]}"
    preds = predict(img_np)

    return img_np, true_label_text, preds


#Gradio
with gr.Blocks(title="CIFAR-10 ResNet Classifier") as demo:
    gr.Markdown("# ResNet Classifier - CIFAR-10")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="input image")
            btn_cifar = gr.Button(
                " pick a random image from CIFAR-10", variant="primary"
            )
            btn_predict = gr.Button("Cllassify Image")
            true_label_output = gr.Textbox(
                label="Ground Truth (CIFAR-10)", interactive=False
            )

        with gr.Column():
            output_labels = gr.Label(
                num_top_classes=10, label="Assigned Probabilities"
            )

    btn_cifar.click( 
        fn=get_random_cifar_image,
        inputs=None,
        outputs=[input_image, true_label_output, output_labels],
    )

    btn_predict.click(  
        fn=predict, inputs=input_image, outputs=output_labels
    )

if __name__ == "__main__":
    demo.launch()
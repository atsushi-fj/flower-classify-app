import gradio as gr
import os
import torch

from create_model import create_cct_model
from timeit import default_timer as timer
from model.utils import get_lines


class_names = get_lines("class_names.txt")

cct_model, cct_transforms = create_cct_model()

cct_model.load_state_dict(
    torch.load(f="checkpoint_cct_model.pth",
               map_location=torch.device("cpu"))
)


def predict(img):
    start_time = timer()
    img = cct_transforms(img).unsqueeze(0)
    cct_model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(cct_model(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    end_time = timer()
    pred_time = round(end_time - start_time, 4)
    
    return pred_labels_and_probs, pred_time


title = "Flower Classify App 👁️🌺"
description = "An [CCT-7/7x2] (https://arxiv.org/abs/2104.05704v4)"
article = "Created at my GitHub repo (https://github.com/atsushi-fj/flower-classify-app)"

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

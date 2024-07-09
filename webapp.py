import gradio as gr
from PIL import Image
import numpy as np

def show_image(img):

    # Convert To PIL Image
    image = Image.open(img)
    # image = image.resize((320, 64))
    print(type(image))

    # Convert the image to a NumPy array
    image_array = np.array(image)
    print(type(image_array))

    return image_array, image

app = gr.Interface(
    fn=show_image,
    inputs=gr.Image(label="Input Image Component", type="filepath"),
    outputs=[gr.Image(label="Output Image Component-1", type="filepath"),
             gr.Image(label="Output Image Component-2", type="filepath")
             ]
)

app.launch()

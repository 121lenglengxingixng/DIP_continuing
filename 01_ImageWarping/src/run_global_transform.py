import gradio as gr
import cv2
import numpy as np
import math

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    if flip_horizontal:
        image = np.fliplr(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)

    bag_ori=np.array([pad_size+math.floor(image.shape[0]/2),pad_size+math.floor(image.shape[1]/2)])
    image_ori=np.array([math.floor(image.shape[0]/2),math.floor(image.shape[1]/2)])
    delta_image=[image.shape[0]-2*image_ori[0],image.shape[1]-2*image_ori[1]]

    # translation
    translation_x=math.floor(translation_x)
    translation_y=math.floor(translation_y)

    bag_ori[0]=bag_ori[0]+translation_x
    bag_ori[1]=bag_ori[1]+translation_y
    
    if scale!=1|rotation!=0:
        # scale and rotation
        (h, w) = image.shape[:2]

        # 设置旋转中心为图像的中心
        center = (w // 2, h // 2)

        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, rotation, scale)

        # 计算旋转后图像的边界框大小
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # 应用仿射变换，生成新的旋转图像
        image = cv2.warpAffine(image, M, (new_w, new_h))
        image[np.where((image == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

        image=np.array(image)

        image_ori=np.array([math.floor(image.shape[0]/2),math.floor(image.shape[1]/2)])
        delta_image=[image.shape[0]-2*image_ori[0],image.shape[1]-2*image_ori[1]]


    image_new[bag_ori[0]-image_ori[0]:bag_ori[0]+delta_image[0]+image_ori[0],bag_ori[1]-image_ori[1]:bag_ori[1]+delta_image[1]+image_ori[1],:]=image
    # image = np.array(image_new)
    transformed_image = np.array(image_new)

    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-600, maximum=600, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-600, maximum=600, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()

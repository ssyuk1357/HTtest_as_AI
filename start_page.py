import gradio as gr
from PIL import Image
import numpy as np
import webbrowser
import threading
from torchvision import transforms
from ultralytics import YOLO
from HTtest_in_gradio import app
    
def image_yolo(image):
    global image_feature
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    model = YOLO("best.pt")
    
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)
    
    results = model(img)
    
    label_mapping = {
        0: '집전체', 1: '지붕', 2: '집벽', 3: '문', 4: '창문', 5: '굴뚝', 6: '연기',
        7: '울타리', 8: '길', 9: '연못', 10: '산', 11: '나무', 12: '꽃', 13: '잔디',
        14: '태양', 15: '나무전체', 16: '기둥', 17: '수관', 18: '가지', 19: '뿌리',
        20: '나뭇잎', 21: '열매', 22: '그네', 23: '새', 24: '다람쥐', 25: '구름',
        26: '달', 27: '별'
    }
    
    label_count = {key: 0 for key in label_mapping.keys()}
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            label = int(box.cls)
            label_count[label] += 1

    house_features = []
    tree_features = []

    for label_id in range(28):
        if label_id < 15:
            if label_count[label_id] > 0:
                house_features.append(f"- {label_mapping[label_id]}이(가) 발견됨")
            else:
                house_features.append(f"- {label_mapping[label_id]}이(가) 발견되지 않음")
        elif label_id == 15:
            continue  # Skip '나무전체'
        else:
            if label_count[label_id] > 0:
                tree_features.append(f"- {label_mapping[label_id]}이(가) 발견됨")
            else:
                tree_features.append(f"- {label_mapping[label_id]}이(가) 발견되지 않음")

    tree_features_str = "### 나무 그림\n" + "\n".join(tree_features)
    house_features_str = "### 집 그림\n" + "\n".join(house_features)
    image_feature = tree_features_str + "\n" + house_features_str
    
    with open("image_features.txt", "w", encoding="utf-8") as f:
        f.write(image_feature)
    
    img_with_boxes = results[0].plot()  # Get the image with bounding boxes
    img_with_boxes = Image.fromarray(img_with_boxes)
    
    return img_with_boxes

def next_page():
    threading.Thread(target=app.launch()).start()
    webbrowser.open("http://127.0.0.1:7861")

# 첫 번째 페이지 레이아웃
with gr.Blocks() as first_page:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# 이미지 입력 페이지")
            image_input = gr.Image(type="pil")
        with gr.Column():
            output_image = gr.Image()

    display_button = gr.Button("이미지 분석")
    next_button = gr.Button("다음 페이지로 이동")

    display_button.click(fn=image_yolo, inputs=image_input, outputs=output_image)
    next_button.click(fn=next_page, inputs=None, outputs=gr.Textbox())

# Gradio 인터페이스 실행
first_page.launch(server_port=7860)

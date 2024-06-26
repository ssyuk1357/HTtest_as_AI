import gradio as gr
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from openai import OpenAI

# API 키 조심해서 사용, 제출 제외
client = OpenAI(api_key='Your OpenAI API Key')
image_feature = 'hello'  # 전역 변수로 정의

def read_image_features():
    with open("image_features.txt", "r", encoding="utf-8") as f:
        return f.read()

def counseling_bot_chat(message, chat_history):
    global image_feature  # 전역 변수로 사용

    image_feature = read_image_features()

    content = f"""
당신은 심리치료센터의 전문 상담사입니다. 내담자는 현재 나무그림검사와 집그림검사를 실시했습니다. 각 검사의 검사 기준은 다음과 같습니다:

### 나무그림검사
1. 나무
   - 나무가 생략된 경우: 불안정, 자아 부재
   - 나무가 있을 때: 성장, 안정, 존재
3. 뿌리
   - 뿌리가 있을 때: 심리적 안정, 근본과 안정
4. 가지
   - 가지가 5개 이상인 나무: 발전 가능성, 확장력 
5. 잎
   - 잎이 10개 이상일 때: 풍부한 생명력, 희망, 활력
6. 열매
   - 열매가 있을 때: 행복, 만족, 풍요로움 
   - 열매가 많을 때: 풍부한 행복과 만족, 높은 성취감
   - 여러 종류의 과일이 열린 나무: 사랑과 관심을 원하는 경향이 있다.
7. 지면선
   - 지면선이 있는 그림: 현실 인식, 안정감, 현실감
9. 꽃
    - 꽃이 3개 이상 보일 때: 높은 창의성, 만족감, 희열
    - 꽃이 있을 때: 행복, 만족, 창조성
10. 잔디
    - 잔디가 있을 때: 안정, 일상, 평온
    - 잔디가 3개 이상 있을 때: 강한 안정감, 평온함
11. 태양
    - 태양이 있을 때: 희망, 긍정, 에너지
13. 새
    - 새가 있을 때: 자유, 꿈, 새로운 시작
14. 구름
    - 구름이 있을 때: 상상력, 꿈, 창의성
    - 구름이 많을 때: 감정의 혼란과 불안, 스트레스
15. 달
    - 달이 있을 때: 감정, 여성성, 순환
16. 별
    - 별이 있을 때: 희망, 꿈, 미래
### 집그림검사
1. 문
   - 문이 없는 집: 관계에 대한 회피, 고립, 위축.
   - 문이 있을 때: 내면의 세계에 대한 열린 태도, 소통, 호의
   - 문이 2개 이상: 과도한 소통 욕구, 불안정성
2. 창문
   - 창문이 없는 집: 폐쇄적 사고, 환경에 대한 관심의 결여와 적의.
   - 창문이 많을 때: 혼란과 불안, 과도한 외부 노출
   - 창문의 격자가 2개 이상인 집: 회의감, 외부 세계로부터 자신을 멀리 하려는 것
   - 출입문과 창문이 둘다 없는 집: 성인과 청소년의 그림이라면 임상적으로 병적인 상태
3. 지붕
   - 지붕이 없는 집: 사회불안, 사회적 상황에서 공포나 불안을 경험.
   - 굴뚝이 있을 때: 경계, 보호, 사적인 공간
   - 굴뚝에서 연기가 나는 집: 마음 속에 긴장이 존재하며 가정환경 내에 갈등이나 정서 혼란이 있음
4. 울타리
   - 울타리가 생략된 집: 외부 위협, 취약, 무방비
   - 울타리가 있을 때: 사적인 공간, 경계, 보호
5. 길
   - 길이 생략된 집: 막힘, 방황, 불확실
   - 길이 있을 때: 미래, 발전, 여정
   - 길이 2개 이상 있을 때: 다양한 가능성, 혼란과 방황
6. 산
   - 배경으로 산이 있을 때: 도전, 극복, 힘
7. 꽃
   - 꽃이 있을 때: 행복하고 만족한 상태, 창의성과 , 풍요로움
   - 꽃이 3개 이상일 때: 행복과 만족, 창의성과 풍요로움
8. 잔디
   - 잔디가 있을 때: 일상적인 삶에 대한 만족, 내면의 평온
9. 태양
   - 태양이 생략된 집: 긍정부족, 우울한 기운, 에너지 부족
   - 태양이 있을 때: 자아의 발전, 긍정적인 전망
   - 태양이 2개 이상 있을 때: 과도한 낙관 주의, 과도한 표현
10. 벽
   - 벽을 생략된 집: 현실검증력 상실, 정신분열
   - 벽이 있을 때: 개인공간, 보호, 안전

이러한 기준을 학습하고, 내담자의 검사 결과에 알맞는 상담을 제공해주세요.

현재 내담자의 그림은 다음과 같은 특징을 가지고 있습니다:
{image_feature}

이 정보를 바탕으로 내담자의 그림을 분석하고 적절한 상담을 제공해주세요.
"""
    print(content)

    if message == "":
        return "", chat_history
    else:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": message}
            ]
        )

    chat_history.append([message, completion.choices[0].message.content])
    return "", chat_history

def counseling_bot_undo(message, chat_history):
    if len(chat_history) > 1:
        chat_history.pop()
    return chat_history

def counseling_bot_reset(chat_history):
    chat_history = [[None, "안녕하세요, 상담을 도와드리겠습니다."]]
    return chat_history

with gr.Blocks(theme=gr.themes.Soft()) as app:
    with gr.Tab("상담봇"):
        gr.Markdown(
            value="""
            # <center>상담봇</center>
            <center>안녕하세요. 상담봇입니다. 불편하신 곳이 있을까요?</center>
            """
        )
        cb_chatbot = gr.Chatbot(
            value=[[None, "안녕하세요 상담봇 입니다. 성함을 알려주실수 있나요?."]],
            show_label=False
        )
        # 채팅 다음줄
        with gr.Row():
            cb_user_input = gr.Text(
                lines=1,
                placeholder="입력 창",
                container=False,
                scale=9
            )
            cb_send_btn = gr.Button(
                value="보내기",
                scale=1,
                variant="primary",
                icon="https://cdn-icons-png.flaticon.com/128/12439/12439334.png"
            )
        # 채팅 다다음 줄
        with gr.Row():
            gr.Button(value="되돌리기").click(fn=counseling_bot_undo, inputs=cb_chatbot, outputs=cb_chatbot)
            gr.Button(value="초기화").click(fn=counseling_bot_reset, inputs=cb_chatbot, outputs=cb_chatbot)
            cb_send_btn.click(fn=counseling_bot_chat, inputs=[cb_user_input, cb_chatbot], outputs=[cb_user_input, cb_chatbot])
            cb_user_input.submit(fn=counseling_bot_chat, inputs=[cb_user_input, cb_chatbot], outputs=[cb_user_input, cb_chatbot])

app.launch(server_port=7861)

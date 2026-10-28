import streamlit as st
import torch
from torch import nn
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Digit Recognizer", page_icon="🧠")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_model():
    model = NeuralNetwork()
    state = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

st.title("🧠 AI Digit Recognizer")
st.write("Нарисуй цифру от 0 до 9, и нейросеть попробует её распознать.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Не удалось загрузить модель: {e}")
    st.stop()

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="digit_canvas"
)

if st.button("🔍 Распознать цифру"):
    if canvas_result.image_data is None:
        st.warning("Сначала нарисуй цифру.")
    else:
        arr = canvas_result.image_data.astype(np.uint8)
        image = Image.fromarray(arr).convert("L")
        image = image.resize((28, 28))

        pixels = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            scores = model(tensor)
            probabilities = torch.softmax(scores, dim=1)[0]

        digit = int(probabilities.argmax().item())
        confidence = float(probabilities[digit].item() * 100)

        st.success(f"Нейросеть распознала цифру: {digit}")
        st.metric("Оценка уверенности", f"{confidence:.2f}%")

        st.subheader("Оценки для всех цифр")
        chart_data = {
            "Цифра": list(range(10)),
            "Оценка (%)": (
                probabilities.numpy() * 100
            )
        }
        st.bar_chart(chart_data, x="Цифра", y="Оценка (%)")

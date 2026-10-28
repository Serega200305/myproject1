import streamlit as st
import torch
from torch import nn
import numpy as np
from PIL import Image
import streamlit.components.v1 as components

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
st.write("Нарисуй цифру на чёрном поле.")

components.html("""
<div style="text-align:center">
  <canvas id="pad" width="280" height="280"
    style="width:280px;height:280px;background:black;
    border:2px solid #60a5fa;border-radius:12px;
    touch-action:none;cursor:crosshair"></canvas>
  <br><br>
  <button onclick="clearPad()" style="padding:10px 20px">
    Очистить
  </button>
</div>
<script>
const canvas = document.getElementById('pad');
const ctx = canvas.getContext('2d');
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, 280, 280);
ctx.strokeStyle = 'white';
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
let drawing = false;

function pos(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX-r.left)*canvas.width/r.width,
    y: (e.clientY-r.top)*canvas.height/r.height
  };
}
canvas.onpointerdown = e => {
  drawing = true;
  canvas.setPointerCapture(e.pointerId);
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(p.x,p.y);
};
canvas.onpointermove = e => {
  if (!drawing) return;
  const p = pos(e);
  ctx.lineTo(p.x,p.y);
  ctx.stroke();
};
canvas.onpointerup = () => drawing = false;
canvas.onpointercancel = () => drawing = false;
function clearPad() {
  ctx.fillStyle = 'black';
  ctx.fillRect(0,0,280,280);
}
</script>
""", height=340)

st.info(
    "Ниже загрузи изображение своей цифры: "
    "это позволит передать рисунок из браузера в нейросеть."
)

uploaded = st.file_uploader(
    "Загрузи рисунок цифры в PNG",
    type=["png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("L").resize((28, 28))
    pixels = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)

    model = load_model()

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    digit = int(probabilities.argmax().item())
    st.success(f"Предсказание: {digit}")
    st.metric(
        "Оценка уверенности",
        f"{probabilities[digit].item() * 100:.2f}%"
    )
    st.bar_chart({
        "Оценка (%)": probabilities.numpy() * 100
    })

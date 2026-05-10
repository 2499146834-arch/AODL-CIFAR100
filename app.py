"""CIFAR-100 inference web app."""
import io
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template_string, request, jsonify
from torchvision import transforms
from model import CIFAR100ResNet
from config import DEVICE, IMAGE_SIZE, CIFAR100_MEAN, CIFAR100_STD, CHECKPOINT_DIR

app = Flask(__name__)

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

CKPT = CHECKPOINT_DIR / "exp_loss_LabelSmoothing_0.1_s42_best.pth"
model = CIFAR100ResNet().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Model loaded: {CKPT}")

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CIFAR-100 图像分类</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Microsoft YaHei','PingFang SC',sans-serif;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);min-height:100vh;display:flex;justify-content:center;align-items:center;color:#e0e0e0}
.container{max-width:700px;width:95%;padding:40px;background:rgba(255,255,255,0.06);border-radius:20px;backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.1);box-shadow:0 20px 60px rgba(0,0,0,0.3)}
h1{text-align:center;font-size:28px;margin-bottom:8px;background:linear-gradient(90deg,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{text-align:center;font-size:13px;color:#888;margin-bottom:30px}
.upload-zone{position:relative;border:2px dashed rgba(255,255,255,0.2);border-radius:16px;padding:50px 20px;text-align:center;cursor:pointer;transition:all .3s;overflow:hidden}
.upload-zone:hover,.upload-zone.drag{border-color:#a78bfa;background:rgba(167,139,250,0.08)}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-zone .icon{font-size:48px;margin-bottom:12px}
.upload-zone .hint{font-size:14px;color:#888}
#preview{display:none;text-align:center;margin:20px 0}
#preview img{max-width:200px;max-height:200px;border-radius:12px;border:2px solid rgba(255,255,255,0.15)}
#results{display:none;margin-top:24px}
.result-bar{margin:8px 0;display:flex;align-items:center;gap:12px}
.result-bar .label{width:130px;text-align:right;font-size:13px;color:#ccc;flex-shrink:0}
.result-bar .bar-track{flex:1;height:28px;background:rgba(255,255,255,0.06);border-radius:14px;overflow:hidden;position:relative}
.result-bar .bar-fill{height:100%;border-radius:14px;transition:width .6s ease;display:flex;align-items:center;padding-left:12px;font-size:12px;font-weight:600;color:#fff}
.result-bar .conf{width:55px;font-size:13px;color:#aaa;text-align:left}
.bar-0{background:linear-gradient(90deg,#818cf8,#6366f1)}
.bar-1{background:linear-gradient(90deg,#7dd3fc,#38bdf8)}
.bar-2{background:linear-gradient(90deg,#86efac,#4ade80)}
.bar-3{background:linear-gradient(90deg,#fde68a,#facc15)}
.bar-4{background:linear-gradient(90deg,#fdba74,#f97316)}
#loading{display:none;text-align:center;margin:20px 0}
.spinner{width:36px;height:36px;border:3px solid rgba(255,255,255,0.15);border-top-color:#a78bfa;border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 10px}
@keyframes spin{to{transform:rotate(360deg)}}
.footer{text-align:center;margin-top:30px;font-size:11px;color:#555}
.bottom-badge{display:inline-block;margin-top:4px;padding:2px 10px;background:rgba(167,139,250,0.15);border-radius:10px;color:#a78bfa;font-size:11px}
</style>
</head>
<body>
<div class="container">
  <h1>CIFAR-100 图像分类</h1>
  <p class="subtitle">ResNet-18 + SE + Label Smoothing &bull; 准确率 75.66%</p>
  <div class="upload-zone" id="zone">
    <div class="icon">&#128206;</div>
    <div class="hint">拖拽图片到此处 或 点击选择<br><small>支持 JPG / PNG / WebP</small></div>
    <input type="file" id="fileInput" accept="image/*">
  </div>
  <div id="preview"><img id="previewImg" src=""></div>
  <div id="loading"><div class="spinner"></div><p>正在分析...</p></div>
  <div id="results"></div>
  <div class="footer">
    AODL Project Reproduction &bull; PyTorch + Flask &bull; RTX 5060 Ti
    <br><span class="bottom-badge">75.66% test accuracy on CIFAR-100</span>
  </div>
</div>
<script>
const zone=document.getElementById('zone'),input=document.getElementById('fileInput');
const preview=document.getElementById('preview'),previewImg=document.getElementById('previewImg');
const loading=document.getElementById('loading'),results=document.getElementById('results');
['dragenter','dragover'].forEach(e=>zone.addEventListener(e,e=>{e.preventDefault();zone.classList.add('drag')}));
['dragleave','drop'].forEach(e=>zone.addEventListener(e,e=>{e.preventDefault();zone.classList.remove('drag')}));
zone.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)handle(f)});
input.addEventListener('change',e=>{const f=e.target.files[0];if(f)handle(f)});
function handle(file){
  if(!file.type.startsWith('image/')){alert('请上传图片文件');return}
  const reader=new FileReader();
  reader.onload=e=>{previewImg.src=e.target.result;preview.style.display='block'};
  reader.readAsDataURL(file);
  const fd=new FormData();fd.append('image',file);
  results.style.display='none';loading.style.display='block';
  fetch('/predict',{method:'POST',body:fd}).then(r=>r.json()).then(data=>{
    loading.style.display='none';results.style.display='block';
    let html='';
    data.forEach((p,i)=>{
      html+=`<div class="result-bar">
        <span class="label">${p.class.replace(/_/g,' ')}</span>
        <div class="bar-track"><div class="bar-fill bar-${i}" style="width:${p.conf}%">${p.conf}%</div></div>
        <span class="conf">${p.conf}%</span>
      </div>`;
    });
    results.innerHTML=html;
  }).catch(err=>{loading.style.display='none';alert('分析失败: '+err)});
}
</script>
</body>
</html>"""

@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    top5 = probs.topk(5)
    return jsonify([
        {"class": CIFAR100_CLASSES[idx], "conf": round(conf.item() * 100, 1)}
        for idx, conf in zip(top5.indices, top5.values)
    ])

if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    print("http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

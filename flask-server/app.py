from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict',  methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})






import model_util as mu
import predictor

model = mu.getModel('test10')

first = 'SK㈜와 SK이노베이션은 15일 “테라파워에 2억5000만달러(3270억원)를 투자하기로 했다”고 밝혔다'
second = 'SK그룹은 지난해 6월 확대경영회의 이후 SMR 분야 투자를 검토하기 시작, 지난 5월 테라파워와 포괄적 사업협력을 위한 양해각서(MOU)를 체결했다'

predictor.predict(first, second, model)

first = 'SK㈜와 SK이노베이션은 15일 “테라파워에 2억5000만달러(3270억원)를 투자하기로 했다”고 밝혔다 SK그룹은 지난해 6월 확대경영회의'
second = '이후 SMR 분야 투자를 검토하기 시작, 지난 5월 테라파워와 포괄적 사업협력을 위한 양해각서(MOU)를 체결했다'

predictor.predict(first, second, model)
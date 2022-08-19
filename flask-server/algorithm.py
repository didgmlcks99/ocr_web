import model_util as mu
import predictor
import data

model = mu.getModel('cnn-test10')
ch2idx = data.getCh2idx()

text = '한동훈 법무부 장관이 18일 윤석열 대통령에게\n검찰총장 후보자로 이원석(53·사법연수원 27기) 대검찰청 차장을\n임명 제청했다. 윤 대통령은 이르면 이날 이 차장을 총장 후보자로 지명해 발표할 전망이다.\n이 차장은 전남 보성 출신으로 서울 중동고등학교와 서울대 정치학과를 졸업했다. 그는 1995년\n37회 사법시험에 합격하고 1998년 사법연수원을 수료한 뒤 서울지검 동부지청에서 검사 생활을\n시작했다. 그는 검찰 내 대표적인 특수통으로 분류된다. 수원지검 특수부 검사 시절 당시 대검 검찰연구관으로\n근무하던 윤석열 대통령과 삼성그룹 비자금 및 로비 의혹 사건을 함께 수사했다. 2017년엔 서울중앙지검 특수1부장으로\n국정농단 의혹 사건을 수사하며 박근혜 전 대통령을 직접 조사하고 구속하기도 했다.'

# split from each
replaced = text.replace('\n', ' ')

new_splitted = []

words = replaced.split()

s = 0
for i in range(1, len(words)):
    first = ' '.join(words[s:i])
    second = ' '.join(words[i:len(words)+1])
    
    output = predictor.predict(first, second, model, ch2idx)

    if output == '0':
        s = i
        new_splitted.append(first)

    print(first,end='-->')
    print(second,end='=')
    print(output)
print()

split = ' '.join(words[s:len(words)+1])
new_splitted.append(split)

result = '\n'.join(new_splitted)
print(result)
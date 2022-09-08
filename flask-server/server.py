from flask import Flask, jsonify, request

app = Flask(__name__)

import model_util as mu
import predictor
import data

model = mu.getModel('rnn-test11')
ch2idx = data.getCh2idx()

@app.route("/server/translate", methods=['POST'])
def translate():
    data = request.get_json()
    
    text = data['text']

    replaced = text.replace('\n', ' ')

    new_splitted = []

    words = replaced.split()

    s = 0
    for i in range(1, len(words)):
        first = ' '.join(words[s:i])

        if i < len(words) - 10:
            second = ' '.join(words[i:i+5])
        else: second = ' '.join(words[i:len(words)+1])
            
        
        output = predictor.predict(first, second, model, ch2idx)

        if output == '0':
            s = i
            new_splitted.append(first)

        print(first,end='-->')
        print(second,end='=')
        # print(output)
    print()

    split = ' '.join(words[s:len(words)+1])
    new_splitted.append(split)

    result = '\n'.join(new_splitted)
    print(result)

    # print(output)

    return result

if __name__ == "__main__":
    app.run(debug=True)

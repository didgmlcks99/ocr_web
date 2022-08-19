import torch
import torch.nn.functional as F
import utils
import data

def predict(first, second, model, ch2idx):

    first_syll = utils.split_syll(first)
    second_syll = utils.split_syll(second)

    first = utils.split(first_syll)
    second = utils.split(second_syll)

    max_len = max(len(first), len(second))

    first = utils.syll_enc(first, 1, max_len, ch2idx)
    second = utils.syll_enc(second, 0, max_len, ch2idx)

    first = torch.tensor(first).unsqueeze(dim=0)
    second = torch.tensor(second).unsqueeze(dim=0)

    model.cpu()

    logits = model.forward(first, second)
    probs = F.softmax(logits, dim=1).squeeze(dim=0)

    result = probs[0]*100

    if result >= 20.00: output = 1
    else: output = 0

    return str(output)

    # print(f"{probs[0]*100:.2f}% that next line must be omitted.")

    # result = f"{probs[0]*100:.2f}% that next line must be omitted."
    # return  result
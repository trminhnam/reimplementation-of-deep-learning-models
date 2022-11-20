import torch
# from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, source_lang, target_lang, device, max_length=50):
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.lower() for token in sentence.split(' ')]
    else:
        tokens = [token.lower() for token in sentence]

    # print(tokens)

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, source_lang.word2index["<sos>"])

    # Go through each source token and convert to an index
    text_to_indices = [source_lang.word2index.get(token, source_lang.word2index['<unk>']) for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [target_lang.word2index["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == target_lang.word2index["<eos>"]:
            break

    translated_sentence = [target_lang.index2word[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


# def bleu(data, model, source_lang, target_lang, device):
#     targets = []
#     outputs = []

#     for example in data:
#         src = vars(example)["src"]
#         trg = vars(example)["trg"]

#         prediction = translate_sentence(model, src, source_lang, target_lang, device)
#         prediction = prediction[:-1]  # remove <eos> token

#         targets.append([trg])
#         outputs.append(prediction)

#     return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
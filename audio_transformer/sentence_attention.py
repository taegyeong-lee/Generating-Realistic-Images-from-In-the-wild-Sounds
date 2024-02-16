from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np

class SentenceAttention():
    def __init__(self):
        self.tagger = SequenceTagger.load("flair/pos-english")

    def get_attention(self, text):
        sentence = Sentence(text)
        self.tagger.predict(sentence, return_probabilities_for_all_classes=True)

        nn_attention = []
        nns_attention = []

        for i in range(1, len(text.split(' ')) + 1):
            for item in sentence.get_token(i).tags_proba_dist['pos']:
                if item.value == "NN":
                    nn_attention.append(item.score)
                if item.value == "NNS":
                    nns_attention.append(item.score)
        result = np.array(nn_attention) + np.array(nns_attention)

        result_attention = []
        for idx, item in enumerate(result):
            result_attention.append(float(item))

        return result_attention


if __name__ == "__main__":
    sentence_attention = SentenceAttention()
    attention = sentence_attention.get_attention('a birds are chriping')
    print(attention)

import warnings
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math

warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)  # FutureWarning 제거


class SentenceModel(torch.nn.Module):
    def __init__(self):
        super(SentenceModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('sentence-transformers/all-distilroberta-v1').to(self.device)
        self.row_list = list()

    def cal_sentence_sim(self, predict_sentences, ground_sentences):
        p_sentences = self.model.encode(predict_sentences)
        g_sentences = self.model.encode(ground_sentences)
        cosine_sim = cosine_similarity(p_sentences, g_sentences) * math.exp(0.07)
        return cosine_sim

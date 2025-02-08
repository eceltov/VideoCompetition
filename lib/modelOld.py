import open_clip
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import time

class Model:
    def __init__(self) -> None:
        self.device = "cuda"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                               pretrained='laion2b_s34b_b79k', device=self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        with open('features.pkl', 'rb') as f:
            features = pickle.load(f)
            self.features = features.to(self.device)

        self.image_count = self.features.shape[0]
        self.scores = np.empty(self.image_count)
        self.scores.fill(1 / self.image_count)

    def reset_scores(self):
        self.scores.fill(1 / self.image_count)

    def L2S(self, feature1, feature2):
        res = torch.norm(torch.sub(feature1, feature2))
        return res
    
    def L2SBulk(self, features, feature2):
        res = torch.norm(features - feature2, dim=1)
        return res
    
    def L2SSuperBulk(self, features, subtractors):
        res = torch.norm(features - subtractors, dim=2)
        return res

    def normalize_scores_max(self):
        self.scores /= self.scores.max()

    def update_scores(self, display, likeID, alpha):
        start = time.time()
        display_features = torch.stack([self.features[i] for i in display])
        PFs = torch.exp(-self.L2SBulk(self.features, self.features[likeID]) / alpha)

        display_features_expanded = torch.unsqueeze(display_features, 0)
        display_features_expanded2 = display_features_expanded.expand(self.image_count, -1, -1)
        subtractor = torch.stack([self.features[i] for i in range(self.image_count)])
        subtractor = torch.unsqueeze(subtractor, 1)

        preNorm = time.time()
        print("pre norm")
        norm = self.L2SSuperBulk(display_features_expanded2, subtractor) / alpha

        preExp = time.time()
        print("pre exp")
        exp = torch.exp(-norm)
        a = exp.sum(1)

        preCount = time.time()
        print("pre count")

        self.scores = (torch.tensor(self.scores).to(self.device) * PFs / a).to("cpu")

        #for i in range(self.image_count):
            #NF = torch.exp(-self.L2SBulk(display_features, self.features[i]) / alpha).sum()
            #NF = a[i]
            #self.scores[i] = self.scores[i] * PFs[i] / a[i]
        #print(self.scores)
        #print(scores)
        preNorm = time.time()
        print("pre norm")
        self.normalize_scores_max()
        finished = time.time()
        print("finished")
        print(preNorm - start, preExp - preNorm, preCount - preExp, preNorm - preCount, finished - preNorm, finished - start)

    def get_top_score_indices(self, count):
        top_k = np.argsort(-self.scores)
        return top_k[:count]



    def search_clip(self, text: str) -> list[int]:
        """
        The returned indices are indices of the corresponding images in sorted(os.listdir('data'))
        """
        query = self.tokenizer(text).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(query)

            similarities = 1 - (F.normalize(text_features) @ F.normalize(self.features).T)

            sorted_indices = torch.argsort(similarities)[0]

        return sorted_indices

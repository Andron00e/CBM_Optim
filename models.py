import clip
import open_clip
from configs import *
from sentence_transformers import SentenceTransformer

class DownloadCLIP:
    def __init__(self,
                 name: str,
                 author: str,
                 device):
        self.name = name
        self.author = author
        self.device = device

    def load(self):
        clip, _, preprocess = open_clip.create_model_and_transforms(self.name, pretrained=self.author, device=self.device)
        return clip, preprocess

class DownloadCLIPhuggingface:
    def __init__(self,
                 name: str,
                 author: str,
                 device):
        super(DownloadCLIP).__init__()

    def load(self):
        pass

class ConceptNetFiltering:
    """
    Class which provides concept generation via ConceptNet api
    Args:
         parent_list: list based on which  set of concepts is created
    """
    def __init__(self, parent_list: list):
        self.parent_list = parent_list
        self.limit = 200
        self.relations = ["HasA", "IsA", "PartOf", "HasProperty", "MadeOf", "AtLocation"]
        self.max_len = 30
        self.print_prob = 0.2
        self.class_sim_cutoff = 0.85
        self.other_sim_cutoff = 0.9

    def get_init_conceptnet(self, parent_list, limit=200,
                            relations=["HasA", "IsA", "PartOf", "HasProperty", "MadeOf", "AtLocation"]):
        concepts = set()

        for cls in tqdm(parent_list):
            words = cls.replace(',', '').split(' ')
            for word in words:
                obj = requests.get('http://api.conceptnet.io/c/en/{}?limit={}'.format(word, limit)).json()
                obj.keys()
                for dicti in obj['edges']:
                    rel = dicti['rel']['label']
                    try:
                        if dicti['start']['language'] != 'en' or dicti['end']['language'] != 'en':
                            continue
                    except(KeyError):
                        continue

                    if rel in relations:
                        if rel in ["IsA"]:
                            concepts.add(dicti['end']['label'])
                        else:
                            concepts.add(dicti['start']['label'])
                            concepts.add(dicti['end']['label'])
        return concepts

    def _clip_dot_prods(self, list1, list2, device="cuda", clip_name="ViT-B/16", batch_size=500):
        "Returns: numpy array with dot products"
        clip_model, _ = clip.load(clip_name, device=device)
        text1 = clip.tokenize(list1).to(device)
        text2 = clip.tokenize(list2).to(device)

        features1 = []
        with torch.no_grad():
            for i in range(math.ceil(len(text1)/batch_size)):
                features1.append(clip_model.encode_text(text1[batch_size*i:batch_size*(i+1)]))
            features1 = torch.cat(features1, dim=0)
            features1 /= features1.norm(dim=1, keepdim=True)

        features2 = []
        with torch.no_grad():
            for i in range(math.ceil(len(text2)/batch_size)):
                features2.append(clip_model.encode_text(text2[batch_size*i:batch_size*(i+1)]))
            features2 = torch.cat(features2, dim=0)
            features2 /= features2.norm(dim=1, keepdim=True)

        dot_prods = features1 @ features2.T
        return dot_prods.cpu().numpy()

    def filter_too_similar_to_cls(self, parent_list, sim_cutoff, device="cuda", print_prob=0):
        # first check simple text matches
        print(len(self.remove_too_long))
        concepts = list(self.remove_too_long)
        concepts = sorted(concepts)

        for cls in parent_list:
            for prefix in ["", "a ", "A ", "an ", "An ", "the ", "The "]:
                try:
                    concepts.remove(prefix + cls)
                    if random.random() < print_prob:
                        print("Class:{} - Deleting {}".format(cls, prefix + cls))
                except(ValueError):
                    pass
            try:
                concepts.remove(cls.upper())
            except(ValueError):
                pass
            try:
                concepts.remove(cls[0].upper() + cls[1:])
            except(ValueError):
                pass
        print(len(concepts))

        mpnet_model = SentenceTransformer('all-mpnet-base-v2')
        class_features_m = mpnet_model.encode(parent_list)
        concept_features_m = mpnet_model.encode(concepts)
        dot_prods_m = class_features_m @ concept_features_m.T
        dot_prods_c = self._clip_dot_prods(parent_list, concepts)
        # weighted since mpnet has higher variance
        dot_prods = (dot_prods_m + 3 * dot_prods_c) / 4

        to_delete = []
        for i in range(len(parent_list)):
            for j in range(len(concepts)):
                prod = dot_prods[i, j]
                if prod >= sim_cutoff and i != j:
                    if j not in to_delete:
                        to_delete.append(j)
                        if random.random() < print_prob:
                            print("Class:{} - Concept:{}, sim:{:.3f} - Deleting {}".format(parent_list[i], concepts[j],
                                                                                           dot_prods[i, j],
                                                                                           concepts[j]))
                            print("".format(concepts[j]))

        to_delete = sorted(to_delete)[::-1]

        for item in to_delete:
            concepts.pop(item)
        print(len(concepts))
        return concepts

    def filter_too_similar(self, sim_cutoff, device="cuda", print_prob=0):

        concepts = self.filter_too_similar_to_cls
        mpnet_model = SentenceTransformer('all-mpnet-base-v2')
        concept_features = mpnet_model.encode(concepts)

        dot_prods_m = concept_features @ concept_features.T
        dot_prods_c = self._clip_dot_prods(concepts, concepts)

        dot_prods = (dot_prods_m + 3 * dot_prods_c) / 4

        to_delete = []
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                prod = dot_prods[i, j]
                if prod >= sim_cutoff and i != j:
                    if i not in to_delete and j not in to_delete:
                        to_print = random.random() < print_prob
                        # Deletes the concept with lower average similarity to other concepts - idea is to keep more general concepts
                        if np.sum(dot_prods[i]) < np.sum(dot_prods[j]):
                            to_delete.append(i)
                            if to_print:
                                print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j],
                                                                                  dot_prods[i, j], concepts[i]))
                        else:
                            to_delete.append(j)
                            if to_print:
                                print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j],
                                                                                  dot_prods[i, j], concepts[j]))

        to_delete = sorted(to_delete)[::-1]
        for item in to_delete:
            concepts.pop(item)
        print(len(concepts))
        return concepts

    def remove_too_long(self, max_len, print_prob=0):

        concepts = self.get_init_conceptnet
        new_concepts = []
        for concept in concepts:
            if len(concept) <= max_len:
                new_concepts.append(concept)
            else:
                if random.random() < print_prob:
                    print(len(concept), concept)
        print(len(concepts), len(new_concepts))
        return new_concepts

    def create(self):

        concepts = self.get_init_conceptnet(parent_list=self.parent_list, limit=self.limit, relations=self.relations)
        concepts = self.remove_too_long(concepts, max_len=self.max_len, print_prob=self.print_prob)
        concepts = self.filter_too_similar_to_cls(concepts, self.parent_list, self.class_sim_cutoff, print_prob=self.print_prob)
        concepts = self.filter_too_similar(concepts, other_sim_cutoff=self.other_sim_cutoff, print_prob=self.print_prob)

        return concepts

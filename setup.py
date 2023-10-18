from configs import *
#from dataset_utils import classes TODO: realize import from dataset_utils
from sklearn.manifold import TSNE
from bokeh.io import output_notebook
from sklearn.decomposition import PCA
import bokeh.models as bm, bokeh.plotting as pl
from sklearn.preprocessing import StandardScaler

def remove_prefixes(strings):
    prefixes = ['a', 'an', 'the']
    result = []

    for string in strings:
        words = string.split()
        if words[0].lower() in prefixes:
            result.append(' '.join(words[1:]))
        else:
            result.append(string)

    return result

with open("all_concepts.txt", "r") as f:
    concepts = f.read().lower().split("\n")
    concepts = remove_prefixes(concepts)

def similarity(a: torch.Tensor, b: torch.Tensor):
    nom = a @ b.T
    denom = a.norm(dim=-1) * b.norm(dim=-1)
    return nom / denom

def cubed_similarity(a: torch.Tensor, b: torch.Tensor):
    nom = a**3 @ (b**3).T
    denom = (a**3).norm(dim=-1) * (b**3).norm(dim=-1)
    return nom / denom

def get_dot_prods_matrix(image_features: torch.Tensor, text_features: torch.Tensor):
    matrix = torch.zeros((len(image_features), len(text_features)))
    for i, im_vector in enumerate(image_features):
        for j, text_vector in enumerate(text_features):
            matrix[i][j] = similarity(im_vector, text_vector)


def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxiliary info on hover """
    if isinstance(color, str): color = [color] * len(x)
    data_source = bm.ColumnDataSource({'x': x, 'y': y, 'color': color, **kwargs})

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show: pl.show(fig)
    return fig


def draw_similarity_scores(scores_dict, true_class):
    """
    scores_dict: a nested dictionary with similarity scores
    true_class: the true image class for which scores should be plotted
    """

    if true_class not in scores_dict:
        print(f"True class '{true_class}' not found in the scores dictionary.")
        return

    scores = scores_dict[true_class]
    df = pd.DataFrame(list(scores.items()), columns=['Class', 'Total Similarity Score'])

    plt.figure(figsize=(6, 3))  # 12 6
    sns.scatterplot(data=df, x='Class', y='Total Similarity Score')
    plt.title(f"Similarity Scores for True Class: {true_class}")
    plt.xticks(rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Total Similarity Score")
    plt.tight_layout()
    plt.show()

class EmbeddingExperiments:
    def __init__(self,

                 ):
        pass
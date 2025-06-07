from typing import Optional, Any
from numpy import ndarray, asarray, dtype, inner
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch import Tensor
from typing import List, Callable
from numpy.linalg import norm

class Pipeline():
    def __init__(self):
        self.classifier = None

    def fit(self, input: ndarray[float], output: Optional[ndarray[int]]) -> None:
        real_input = []
        real_output = []
        for i, o in zip(input, output):
            if i is not None:
                real_input.append(i)
                real_output.append(o)

        real_input = asarray(real_input)
        real_output = asarray(real_output)

        self.classifier.fit(real_input, real_output)

    def predict(self, input: List[float]) -> ndarray[float]:
        real_input = []
        missing = []

        for i, embedd in enumerate(input):
            if embedd is None:
                missing.append(i)
            else:
                real_input.append(embedd)

        pred_out = []
        if len(real_input) > 0:
            prediction = self.classifier.predict_proba(real_input)
            if prediction.ndim == 2:
                prediction = prediction[:, 1]
        else:
            prediction = []

        current_ind = 0
        for i in range(len(input)):
            if i in missing:
                pred_out.append(0)
            else:
                pred_out.append(prediction[current_ind])
                current_ind += 1

        return asarray(pred_out)

    def classify(self, input: List[float], max_t: float) -> ndarray[int]:
        rounded = []
        for pred in self.predict(input):
            if pred < max_t:
                rounded.append(0)
            else:
                rounded.append(1)

        return asarray(rounded)
class LinkPredictionPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.classifier = LinkPredictionClassifier()


class DownstreamPipeline(Pipeline):
    def __init__(self,
                 classifier_type: str,
                 classifier_kwargs: Optional[dict] = None
                 ):

        super().__init__()
        if classifier_type == "forest":
            self.classifier = BalancedRandomForestClassifier(**classifier_kwargs)
        if classifier_type == "mlp":
            self.classifier = MLPClassifier(**classifier_kwargs)
        if classifier_type == "minusnorm":
            self.classifier = MinusNormClassifier()
        if classifier_type == "inner":
            self.classifier = InnerSimilarityClassifier()
        if classifier_type == "norminner":
            self.classifier = NormInnerSimilarityClassifier()
        if classifier_type == "svm":
            self.classifier = SVC(**classifier_kwargs, probability=True)

class LinkPredictionClassifier():
    def __init__(self):
        self.oligo_embedding = None
        self.interaction_function = None

    def fit(self, input: ndarray[float], output: Optional[ndarray[int]]):
        pass

    def predict_proba(self, input: ndarray[float]) -> ndarray[Any, dtype[Any]]:
        output = []

        for pair in input:
            val1 = self.interaction_function(h=Tensor([pair[0]]),r=Tensor([self.oligo_embedding]),t=Tensor([pair[1]]))[0]
            val2 = self.interaction_function(h=Tensor([pair[1]]),r=Tensor([self.oligo_embedding]),t=Tensor([pair[0]]))[0]
            output.append(val1)
            output.append(val2)

        return asarray(output)

class SimilarityClassifier():
    similarity_function: Callable[[ndarray,ndarray],float]

    def fit(self, input: ndarray[float], output: Optional[ndarray[int]] = None):
        pass

    def predict_proba(self, input: ndarray[float]):
        output = []

        for pair in input:
            output.append(self.similarity_function(pair[0],pair[1]))

        return asarray(output)

class InnerSimilarityClassifier(SimilarityClassifier):
    def __init__(self):
        self.similarity_function = inner

def minus_distance(a: ndarray, b: ndarray):
    return -norm(a-b)

class MinusNormClassifier(SimilarityClassifier):
    def __init__(self):
        self.similarity_function = minus_distance

def norm_inner(a: ndarray, b: ndarray):
    norm_a = norm(a)
    norm_b = norm(b)
    return inner(a,b)/(norm_a*norm_b)

class NormInnerSimilarityClassifier(SimilarityClassifier):
    def __init__(self):
        self.similarity_function = norm_inner
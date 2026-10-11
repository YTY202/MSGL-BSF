import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class DualViewTextEncoder:
    """Parameter-free lexical/semantic encoder for biomedical descriptions.

    Word n-grams capture biomedical phrases while character n-grams retain
    morphology, abbreviations and rare chemical names.  Both views are fitted
    jointly for diseases and metabolites, compressed independently, and
    confidence-weighted per document.
    """

    def __init__(
        self,
        out_dim=64,
        word_features=12000,
        char_features=16000,
        random_state=0,
    ):
        if out_dim % 2 != 0:
            raise ValueError("out_dim must be even")
        self.view_dim = out_dim // 2
        self.random_state = random_state
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            max_features=word_features,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=char_features,
        )

    def _compress(self, matrix):
        max_rank = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
        if max_rank < 1:
            raise ValueError("Not enough text variation to build embeddings")
        rank = min(self.view_dim, max_rank)
        svd = TruncatedSVD(
            n_components=rank,
            n_iter=7,
            random_state=self.random_state,
        )
        reduced = svd.fit_transform(matrix)
        if rank < self.view_dim:
            reduced = np.pad(reduced, ((0, 0), (0, self.view_dim - rank)))
        return normalize(reduced, norm="l2")

    @staticmethod
    def _confidence(matrix):
        # Rich descriptions should have more influence than near-empty ones.
        nonzero = np.maximum(matrix.getnnz(axis=1), 1)
        score = np.log1p(nonzero).astype(np.float32)
        return score

    def fit_transform(self, texts, device=None):
        if not texts:
            raise ValueError("texts must not be empty")
        clean_texts = [str(text).strip() or "[missing]" for text in texts]
        word_matrix = self.word_vectorizer.fit_transform(clean_texts)
        char_matrix = self.char_vectorizer.fit_transform(clean_texts)

        word_embedding = self._compress(word_matrix)
        char_embedding = self._compress(char_matrix)

        confidence = np.stack(
            [self._confidence(word_matrix), self._confidence(char_matrix)],
            axis=1,
        )
        confidence = np.exp(confidence - confidence.max(axis=1, keepdims=True))
        confidence /= confidence.sum(axis=1, keepdims=True)

        embedding = np.concatenate(
            [
                word_embedding * confidence[:, 0:1],
                char_embedding * confidence[:, 1:2],
            ],
            axis=1,
        )
        embedding = normalize(embedding, norm="l2").astype(np.float32)
        return torch.from_numpy(embedding).to(device)

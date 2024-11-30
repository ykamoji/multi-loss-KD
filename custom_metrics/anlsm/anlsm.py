import datasets
import evaluate
from anls import anls_score


_CITATION = """\
@misc{biten2019scenetextvisualquestion,
      title={Scene Text Visual Question Answering}, 
      author={Ali Furkan Biten and Ruben Tito and Andres Mafla and Lluis Gomez and Marçal Rusiñol and Ernest Valveny and C. V. Jawahar and Dimosthenis Karatzas},
      year={2019},
      eprint={1905.13648},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1905.13648}, 
}
"""

_DESCRIPTION = """\
The Average Normalized Levenshtein Similarity (ANLS) proposed by [Biten+ ICCV'19] smoothly captures the OCR mistakes a
pplying a slight penalization in case of correct intended responses, but badly recognized. It also makes use of a 
threshold of value 0.5 that dictates whether the output of the metric will be the ANLS if its value is equal or 
bigger than 0.5 or 0 otherwise. The key point of this threshold is to determine if the answer has been correctly 
selected but not properly recognized, or on the contrary, the output is a wrong text selected from the options and 
given as an answer.
"""

_KWARGS_DESCRIPTION = """
Computes ANLS score of translated segments against one or more references.
Args:
    predictions: list of translations to score.
    references: list of lists of or just a list of references for each translation.
    tokenizer : approach used for tokenizing `predictions` and `references`.
        The default tokenizer is `tokenizer_13a`, a minimal tokenization approach that is equivalent to `mteval-v13a`, used by WMT.
        This can be replaced by any function that takes a string as input and returns a list of tokens as output.
Returns:
    'anls': anls score,
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Anls(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/shunk031/ANLS"],
            reference_urls=[
                "https://arxiv.org/abs/1905.13648"
            ],
        )

    def _compute(self, predictions, references):
        # if only one reference is provided make sure we still use list of lists
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        avg_score = 0
        for i in range(len(predictions)):
            avg_score += anls_score(prediction=predictions[i], gold_labels=references[i])

        avg_score /= len(predictions)

        return {"anls": f"{avg_score:.10f}"}


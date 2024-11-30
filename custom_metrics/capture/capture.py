import datasets
import evaluate
from capture_metric.capture import CAPTURE


_CITATION = """\
@article{dong2024benchmarking,
  title={Benchmarking and Improving Detail Image Caption},
  author={Dong, Hongyuan and Li, Jiawen and Wu, Bohong and Wang, Jiacong and Zhang, Yuan and Guo, Haoyuan},
  journal={arXiv preprint arXiv:2405.19092},
  year={2024}
}
"""

_DESCRIPTION = """\
Image captioning has long been regarded as a fundamental task in visual understanding. Recently, however, few large 
vision-language model (LVLM) research discusses model's image captioning performance because of the outdated 
short-caption benchmarks and unreliable evaluation metrics. In this work, we propose to benchmark detail image caption 
task by curating high-quality evaluation datasets annotated by human experts, GPT-4V and Gemini-1.5-Pro. We also design 
a more reliable caption evaluation metric called CAPTURE (CAPtion evaluation by exTracting and coUpling coRE 
information). CAPTURE extracts visual elements, e.g., objects, attributes and relations from captions, and 
then matches these elements through three stages, achieving the highest consistency with expert judgements over 
other rule-based or model-based caption metrics. The proposed benchmark and metric provide reliable evaluation 
for LVLM's detailed image captioning ability. Guided by this evaluation, we further explore to unleash LVLM's detail 
caption capabilities by synthesizing high-quality data through a five-stage data construction pipeline. Our pipeline 
only uses a given LVLM itself and other open-source tools, without any human or GPT-4V annotation in the loop. 
Experiments show that the proposed data construction strategy significantly improves model-generated detail caption 
data quality for LVLMs with leading performance, and the data quality can be further improved in a self-looping 
paradigm.
"""

_KWARGS_DESCRIPTION = """
Computes CAPTURE score of translated segments against one or more references.
Args:
    predictions: list of translations to score.
    references: list of lists of or just a list of references for each translation.
    tokenizer : approach used for tokenizing `predictions` and `references`.
        The default tokenizer is `tokenizer_13a`, a minimal tokenization approach that is equivalent to `mteval-v13a`, used by WMT.
        This can be replaced by any function that takes a string as input and returns a list of tokens as output.
Returns:
    'capture': capture score,
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
            codebase_urls=["https://github.com/foundation-multimodal-models/CAPTURE"],
            reference_urls=[
                "https://arxiv.org/abs/2405.19092",
            ],
        )

    def _compute(self, predictions, references):
        # if only one reference is provided make sure we still use list of lists
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        references = {idx:ref for idx, ref in enumerate(references)}
        predictions = {idx:[p] for idx, p in enumerate(predictions)}

        evaluator = CAPTURE(soft_matching=False)

        score = evaluator.compute_score(references, predictions)

        return {"capture": f"{score:.10f}"}


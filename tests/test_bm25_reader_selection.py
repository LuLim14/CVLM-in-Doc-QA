from types import SimpleNamespace

from cvlm_dataset import CvlmTrainDataset
from train_sft import SftFullDocDataset


class _WhitespaceTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(str(text).split())))}


def test_training_selector_is_identical_to_headline_eval_selector():
    tokenizer = _WhitespaceTokenizer()
    train_obj = SimpleNamespace(tokenizer=tokenizer)
    eval_obj = SimpleNamespace(_dec_tok=tokenizer)
    document = (
        "The first sentence concerns cats and gardens. "
        "Paris is the capital of France. "
        "The final sentence concerns oceans and ships."
    )
    question = "What is the capital of France?"
    doc_ids = tokenizer(document)["input_ids"]

    train_ids = SftFullDocDataset._query_selected_ids(
        train_obj, doc_ids, document, question, budget=8
    )
    eval_ids = CvlmTrainDataset._query_selected_ids(
        eval_obj, doc_ids, document, question, budget=8
    )

    assert train_ids == eval_ids
    assert 0 < len(train_ids) <= 8


def test_query_selector_falls_back_to_prefix_when_no_sentence_fits():
    tokenizer = _WhitespaceTokenizer()
    train_obj = SimpleNamespace(tokenizer=tokenizer)
    document = "one two three four. five six seven eight."
    doc_ids = tokenizer(document)["input_ids"]

    selected = SftFullDocDataset._query_selected_ids(
        train_obj, doc_ids, document, "unseen term", budget=2
    )

    assert selected == doc_ids[:2]

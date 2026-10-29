# Dataset + collation for CVLM.
#
# The on-the-fly variant: the dataset yields raw token ids for (source, prompt,
# answer). The model's frozen text encoder embeds `source_ids` at forward time,
# so there is no precomputed embeddings cache anywhere.

from __future__ import annotations

import bisect
import json
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


def compute_chunk_decoder_spans(
    enc_offsets: Sequence[Tuple[int, int]],
    dec_offsets: Sequence[Tuple[int, int]],
    compression_rate: int,
    max_chunks: int,
) -> List[Tuple[int, int]]:
    """Map encoder-token chunks to decoder-token spans via char offsets.

    Chunk i covers encoder tokens [i*cr, (i+1)*cr) — the same chunking as
    _chunked_attention_pool (including the max_chunks==max_vision_len cap), so
    chunk i here corresponds 1:1 to vision token i. Its char span starts at
    the first covered encoder token's char start. Decoder-token boundaries are
    found by bisecting decoder char offsets: boundary b_i = first decoder
    token whose char end exceeds the chunk's char start; the final boundary is
    the first decoder token starting at/after the covered region's char end
    (drops decoder tokens beyond the encoder-truncated document part).

    The mapping is NOT byte-exact at chunk boundaries (encoder and decoder
    tokenizations disagree on word splits); a decoder token straddling a
    boundary is assigned to the last chunk whose char start it covers. Some
    spans may be empty (multiple encoder tokens inside one decoder token) —
    callers keep such chunks compressed.

    Returns a list of (start, end) half-open decoder-token index pairs, one
    per chunk, contiguous and non-decreasing.
    """
    L = len(enc_offsets)
    cr = max(int(compression_rate), 1)
    if L == 0 or len(dec_offsets) == 0:
        return []
    n = min((L + cr - 1) // cr, max(int(max_chunks), 1))
    dec_starts = [s for s, _ in dec_offsets]
    dec_ends = [e for _, e in dec_offsets]
    bounds: List[int] = []
    for i in range(n):
        chunk_char_start = enc_offsets[i * cr][0]
        bounds.append(bisect.bisect_right(dec_ends, chunk_char_start))
    last_enc = min(n * cr, L) - 1
    covered_char_end = enc_offsets[last_enc][1]
    bounds.append(bisect.bisect_left(dec_starts, covered_char_end))
    for i in range(1, len(bounds)):
        if bounds[i] < bounds[i - 1]:
            bounds[i] = bounds[i - 1]
    return [(bounds[i], bounds[i + 1]) for i in range(n)]


class CvlmTrainDataset(Dataset):
    """Raw-text dataset: yields encoder source ids + decoder prompt/answer ids."""

    def __init__(
        self,
        hf_dataset_name: str,
        hf_split: str,
        decoder_tokenizer_name: str,
        encoder_tokenizer_name: str,
        max_prompt_len: int,
        max_answer_len: int,
        max_source_len: int,
        max_samples: int = 0,
        doc_truncation: str = "none",
        doc_budget_cr: int = 0,
        doc_budget_cap: int = 0,
        doc_truncation_seed: int = 42,
        mixed_fidelity_p: float = 0.0,
        mixed_chunk_cr: int = 0,
        mixed_max_chunks: int = 0,
        doc_selection_file: str = "",
    ) -> None:
        self._hf: HFDataset = load_dataset(hf_dataset_name, split=hf_split)
        if max_samples > 0:
            self._hf = self._hf.select(range(min(max_samples, len(self._hf))))
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len
        self.max_source_len = max_source_len

        # Optional eval-only truncation of the document inside `full_prompt_ids`
        # (consumed by the baseline_llm_full / sft eval paths). "none" keeps the
        # historical behaviour: full document + "\n\n" + question. The budgeted
        # strategies (first/last/random/oracle) cut the document to a
        # decoder-token window of size min(ceil(L_enc / doc_budget_cr),
        # doc_budget_cap) — i.e. exactly the vision-token budget CVLM would
        # spend at compression_rate=doc_budget_cr. "file" (skim-then-read)
        # replaces the document with `selected_text` from a
        # tools/score_chunks.py JSONL, matched by idx = position in this
        # filtered dataset; the budget was applied offline by that tool.
        self.doc_truncation = str(doc_truncation)
        self.doc_budget_cr = int(doc_budget_cr)
        self.doc_budget_cap = int(doc_budget_cap)
        self.doc_truncation_seed = int(doc_truncation_seed)
        self.doc_selection_file = str(doc_selection_file or "")
        if self.doc_truncation not in ("none", "first", "last", "random", "oracle", "query", "file"):
            raise ValueError(f"Unknown doc_truncation: {self.doc_truncation!r}")
        if self.doc_truncation == "file":
            if not self.doc_selection_file:
                raise ValueError("doc_truncation='file' requires doc_selection_file (tools/score_chunks.py JSONL)")
        elif self.doc_truncation != "none":
            if self.doc_budget_cr <= 0:
                raise ValueError("doc_truncation != 'none' requires doc_budget_cr > 0")
            if self.doc_budget_cap <= 0:
                raise ValueError("doc_truncation != 'none' requires doc_budget_cap > 0 (max_vision_len)")
            print(
                f"CvlmTrainDataset: doc_truncation={self.doc_truncation}, "
                f"budget=min(ceil(L_enc/{self.doc_budget_cr}), {self.doc_budget_cap}) decoder tokens"
            )
        # idx → truncated document window (decoder-token ids); idx → answer-found
        # flag (oracle strategy only). Memoised so the teacher-forcing and
        # generate passes see identical windows ('random' included) and hits are
        # counted once per sample.
        self._doc_window_cache: Dict[int, List[int]] = {}
        self._oracle_hits: Dict[int, bool] = {}

        self._dec_tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            decoder_tokenizer_name, use_fast=False
        )
        if self._dec_tok.pad_token is None:
            self._dec_tok.pad_token = self._dec_tok.eos_token
        self._enc_tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            encoder_tokenizer_name, use_fast=True, trust_remote_code=True
        )
        if self._enc_tok.pad_token is None:
            self._enc_tok.pad_token = self._enc_tok.eos_token or self._enc_tok.cls_token
        self.dec_pad_id: int = self._dec_tok.pad_token_id
        self.enc_pad_id: int = self._enc_tok.pad_token_id

        # MIXED-FIDELITY extras: when enabled, __getitem__ additionally yields
        # the decoder-tokenised document (tokenised ONCE, with a fast decoder
        # tokenizer for char offsets) and per-chunk decoder-token spans, where
        # chunks are the same [i*cr, (i+1)*cr) encoder-token windows the model's
        # _chunked_attention_pool compresses into vision tokens. Chunking is
        # fixed at construction time (mixed_chunk_cr), so this mode is
        # incompatible with an in-training cr schedule.
        self.mixed_fidelity: bool = float(mixed_fidelity_p) > 0.0
        self.mixed_chunk_cr = int(mixed_chunk_cr)
        self.mixed_max_chunks = int(mixed_max_chunks)
        self._mixed_dec_tok: PreTrainedTokenizer | None = None
        if self.mixed_fidelity:
            if self.mixed_chunk_cr <= 0:
                raise ValueError("mixed_fidelity_p > 0 requires mixed_chunk_cr > 0 (the training compression_rate)")
            if self.mixed_max_chunks <= 0:
                raise ValueError("mixed_fidelity_p > 0 requires mixed_max_chunks > 0 (max_vision_len)")
            if not getattr(self._enc_tok, "is_fast", False):
                raise ValueError("mixed_fidelity_p > 0 requires a FAST encoder tokenizer (char offset mapping)")
            self._mixed_dec_tok = AutoTokenizer.from_pretrained(decoder_tokenizer_name, use_fast=True)
            if not getattr(self._mixed_dec_tok, "is_fast", False):
                raise ValueError("mixed_fidelity_p > 0 requires a FAST decoder tokenizer (char offset mapping)")
            print(
                f"CvlmTrainDataset: MIXED-FIDELITY extras enabled "
                f"(chunk_cr={self.mixed_chunk_cr}, max_chunks={self.mixed_max_chunks})"
            )

        # Vectorised length pass: one batched fast-tokenizer call per field.
        # PwC rows have "input" (source document to compress), "prompt" (the
        # user's question about the document), and "answer".
        inputs = list(self._hf["input"])
        prompts = list(self._hf["prompt"])
        answers = list(self._hf["answer"])

        print(f"Tokenising {len(inputs)} samples to compute length filter...")
        enc_lens = self._batched_lengths(self._enc_tok, inputs)        # source lens in encoder tokens
        dec_prompt_lens = self._batched_lengths(self._dec_tok, prompts)  # question lens in decoder tokens
        dec_answer_lens = self._batched_lengths(self._dec_tok, answers)

        keep = []
        for i in range(len(inputs)):
            if enc_lens[i] <= 0 or enc_lens[i] > max_source_len:
                continue
            if dec_prompt_lens[i] > max_prompt_len:
                continue
            if dec_answer_lens[i] > max_answer_len:
                continue
            keep.append(i)
        self._row_indices: List[int] = keep
        print(
            f"CvlmTrainDataset: kept {len(keep)}/{len(inputs)} samples "
            f"(source_tokens<={max_source_len}, prompt_tokens<={max_prompt_len}, "
            f"answer_tokens<={max_answer_len})"
        )
        if not keep:
            raise RuntimeError("No samples passed length filters; loosen caps.")

        # Skim-then-read selections must be loaded AFTER the filter so the line
        # count can be validated against the filtered length (idx alignment).
        self._doc_selection: Dict[int, str] = {}
        if self.doc_truncation == "file":
            self._load_doc_selection()

    def _load_doc_selection(self) -> None:
        """Load {idx -> selected_text} from a tools/score_chunks.py JSONL.

        idx is the position in THIS filtered dataset; alignment therefore
        requires the scoring run to have used the same dataset/split and the
        same max_source_len / max_prompt_len / max_answer_len / max_samples.
        A line-count mismatch is the tell-tale of diverged filters and fails
        loudly here rather than silently mispairing documents and questions.
        """
        records: List[Dict[str, Any]] = []
        with open(self.doc_selection_file) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{self.doc_selection_file}:{line_no}: invalid JSON: {e}") from e
        n = len(self._row_indices)
        if len(records) != n:
            raise ValueError(
                f"doc_selection_file {self.doc_selection_file!r} has {len(records)} lines but the "
                f"filtered dataset has {n} samples — idx alignment is broken. Re-run "
                f"tools/score_chunks.py with the SAME dataset/split and filter flags "
                f"(max_source_len/max_prompt_len/max_answer_len/max_samples) as this eval."
            )
        for rec in records:
            if "idx" not in rec or "selected_text" not in rec:
                raise ValueError(
                    f"doc_selection_file {self.doc_selection_file!r}: every line needs "
                    f"'idx' and 'selected_text'; got keys {sorted(rec)}"
                )
            i = int(rec["idx"])
            if not 0 <= i < n:
                raise ValueError(f"doc_selection_file idx {i} out of range [0, {n})")
            if i in self._doc_selection:
                raise ValueError(f"doc_selection_file has duplicate idx {i}")
            self._doc_selection[i] = str(rec["selected_text"])
        print(
            f"CvlmTrainDataset: doc_truncation=file — loaded {n} selected_text "
            f"windows from {self.doc_selection_file}"
        )

    @staticmethod
    def _batched_lengths(tokenizer: PreTrainedTokenizer, texts: List[str], batch: int = 1024) -> List[int]:
        out: List[int] = []
        for start in range(0, len(texts), batch):
            enc = tokenizer(
                [str(t) for t in texts[start:start + batch]],
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            out.extend(len(ids) for ids in enc["input_ids"])
        return out

    @staticmethod
    def _find_answer_span(document: str, answer: str) -> tuple[int, int, bool]:
        """Locate the answer inside the document for the `oracle` strategy.

        Tries the exact answer string first, then the first 8-word n-gram of the
        answer that occurs verbatim in the document. Returns
        ``(char_start, char_len, found)``; ``(0, 0, False)`` when nothing matches.
        """
        answer = answer.strip()
        if answer:
            pos = document.find(answer)
            if pos >= 0:
                return pos, len(answer), True
            words = answer.split()
            for i in range(len(words) - 7):
                gram = " ".join(words[i : i + 8])
                pos = document.find(gram)
                if pos >= 0:
                    return pos, len(gram), True
        return 0, 0, False

    def _query_selected_ids(
        self, doc_ids: List[int], document: str, question: str, budget: int
    ) -> List[int]:
        """BM25-style question-conditioned sentence selection within budget."""
        import math
        import re as _re

        sents = [x for x in _re.split(r"(?<=[.!?])\s+", document) if x.strip()]
        if len(sents) <= 1:
            return list(doc_ids[:budget])
        tok = lambda text: _re.findall(r"[a-z0-9]+", text.lower())
        q_terms = set(tok(question))
        sent_terms = [tok(sentence) for sentence in sents]
        df = {term: sum(1 for terms in sent_terms if term in terms) for term in q_terms}
        n_sentences = len(sents)
        avg_len = sum(len(terms) for terms in sent_terms) / max(n_sentences, 1)
        k1, b = 1.5, 0.75
        scores = []
        for terms in sent_terms:
            score = 0.0
            for term in q_terms:
                freq = terms.count(term)
                if not freq:
                    continue
                idf = math.log(
                    1.0 + (n_sentences - df[term] + 0.5) / (df[term] + 0.5)
                )
                score += idf * (freq * (k1 + 1)) / (
                    freq + k1 * (1 - b + b * len(terms) / max(avg_len, 1e-9))
                )
            scores.append(score)
        lengths = [
            len(
                self._dec_tok(
                    sentence, add_special_tokens=False, truncation=False
                )["input_ids"]
            )
            for sentence in sents
        ]
        chosen, used = [], 0
        for sent_idx in sorted(
            range(n_sentences), key=lambda item: (-scores[item], item)
        ):
            if used + lengths[sent_idx] > budget:
                continue
            chosen.append(sent_idx)
            used += lengths[sent_idx]
            if used >= budget:
                break
        if not chosen:
            return list(doc_ids[:budget])
        selected_text = " ".join(sents[item] for item in sorted(chosen))
        selected_ids = self._dec_tok(
            selected_text, add_special_tokens=False, truncation=False
        )["input_ids"]
        return list(selected_ids[:budget])

    def _truncated_doc_ids(self, idx: int, record: Dict[str, Any]) -> List[int]:
        """Decoder-token window of the document under the truncation budget.

        Budget = min(ceil(L_enc / doc_budget_cr), doc_budget_cap) decoder tokens,
        where L_enc is the source length in *encoder* tokens (same measurement
        as the __init__ length filter / compression stats). Window placement:
        first / last / random contiguous (deterministic seed = doc_truncation_seed
        + idx) / oracle (centred on the answer occurrence, first-N fallback).
        The `file` strategy short-circuits: the window is the decoder
        tokenisation of the offline-selected `selected_text` (budget already
        applied by tools/score_chunks.py), no budget formula here.
        """
        cached = self._doc_window_cache.get(idx)
        if cached is not None:
            return cached
        if self.doc_truncation == "file":
            # Skim-then-read: the budgeted window was selected offline by
            # tools/score_chunks.py; just decoder-tokenise its raw text.
            sel = self._doc_selection.get(idx)
            if sel is None:
                raise KeyError(f"doc_truncation='file': no selected_text for idx {idx}")
            window = list(
                self._dec_tok(sel, add_special_tokens=False, truncation=False)["input_ids"]
            )
            self._doc_window_cache[idx] = window
            return window
        document = str(record["input"])
        doc_ids: List[int] = self._dec_tok(
            document, add_special_tokens=False, truncation=False
        )["input_ids"]
        l_enc = min(
            len(self._enc_tok(document, add_special_tokens=False, truncation=False)["input_ids"]),
            self.max_source_len,
        )
        cr = max(self.doc_budget_cr, 1)
        budget = max(min((l_enc + cr - 1) // cr, self.doc_budget_cap), 1)
        n = len(doc_ids)
        start = 0
        if self.doc_truncation == "oracle":
            char_pos, char_len, hit = self._find_answer_span(document, str(record["answer"]))
            self._oracle_hits[idx] = hit
            if hit and n > budget:
                # Char position → token position via prefix re-tokenisation
                # (the slow decoder tokenizer exposes no offset mapping).
                t_start = len(
                    self._dec_tok(document[:char_pos], add_special_tokens=False, truncation=False)["input_ids"]
                )
                t_len = len(
                    self._dec_tok(
                        document[char_pos : char_pos + char_len], add_special_tokens=False, truncation=False
                    )["input_ids"]
                )
                center = t_start + t_len // 2
                start = min(max(center - budget // 2, 0), n - budget)
        elif self.doc_truncation == "query":
            window = self._query_selected_ids(
                doc_ids, document, str(record["prompt"]), budget
            )
            self._doc_window_cache[idx] = window
            return window
        elif n > budget:
            if self.doc_truncation == "last":
                start = n - budget
            elif self.doc_truncation == "random":
                rng = np.random.default_rng(self.doc_truncation_seed + idx)
                start = int(rng.integers(0, n - budget + 1))
            # "first" keeps start = 0.
        window = list(doc_ids[start : start + budget])
        self._doc_window_cache[idx] = window
        return window

    def __len__(self) -> int:
        return len(self._row_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_id = self._row_indices[idx]
        record = self._hf[row_id]
        src = self._enc_tok(
            str(record["input"]),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_source_len,
            return_tensors=None,
            return_offsets_mapping=self.mixed_fidelity,
        ) # TODO: is it good that we use truncated? But we use from _row_indices that prepered from __init__
        # Prompt is the user's *question* about the document, not the document
        # itself. The document goes into `source_ids` (to be compressed into
        # vision tokens); the question stays as raw prompt tokens.
        prompt = self._dec_tok(
            str(record["prompt"]), add_special_tokens=False, truncation=False
        )
        answer = self._dec_tok(
            str(record["answer"]), add_special_tokens=False, truncation=False
        )
        # full_prompt_ids = decoder-tokenised(document + question). Only used by
        # the eval `baseline_llm_full` oracle (upper-bound LLM that sees the raw
        # uncompressed document). Training does not consume this field. With
        # doc_truncation != "none" the document part is cut to a budgeted token
        # window first, then glued to "\n\n" + question as usual.
        if self.doc_truncation != "none":
            doc_window = self._truncated_doc_ids(idx, record)
            tail = self._dec_tok(
                "\n\n" + str(record["prompt"]), add_special_tokens=False, truncation=False
            )
            full_ids = list(doc_window) + list(tail["input_ids"])
        else:
            full_prompt_text = str(record["input"]) + "\n\n" + str(record["prompt"])
            full_prompt = self._dec_tok(
                full_prompt_text, add_special_tokens=False, truncation=False
            )
            full_ids = full_prompt["input_ids"]
        out = {
            "source_ids": torch.as_tensor(src["input_ids"], dtype=torch.long),
            "prompt_ids": torch.as_tensor(prompt["input_ids"], dtype=torch.long),
            "full_prompt_ids": torch.as_tensor(full_ids, dtype=torch.long),
            "answer_ids": torch.as_tensor(answer["input_ids"], dtype=torch.long),
        }
        if self.mixed_fidelity:
            # Decoder-tokenise the document ONCE (fast tokenizer, with char
            # offsets); chunk spans index into these ids. Ids come from the
            # fast tokenizer so ids and spans are internally consistent even if
            # the slow decoder tokenizer would split slightly differently.
            assert self._mixed_dec_tok is not None
            doc_dec = self._mixed_dec_tok(
                str(record["input"]),
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            spans = compute_chunk_decoder_spans(
                src["offset_mapping"],
                doc_dec["offset_mapping"],
                compression_rate=self.mixed_chunk_cr,
                max_chunks=self.mixed_max_chunks,
            )
            out["doc_dec_ids"] = torch.as_tensor(doc_dec["input_ids"], dtype=torch.long)
            out["chunk_dec_spans"] = torch.as_tensor(spans, dtype=torch.long).reshape(-1, 2)
        return out


def make_collate_fn(
    dec_pad_id: int,
    enc_pad_id: int,
) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    def collate_cvlm_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        b = len(samples)

        # Source (encoder input): right-pad with encoder pad id.
        max_s = max(s["source_ids"].shape[0] for s in samples)
        source_ids = torch.full((b, max_s), enc_pad_id, dtype=torch.long)
        source_attention_mask = torch.zeros(b, max_s, dtype=torch.long)

        # Prompt: LEFT-pad so real tokens sit flush against the vision block.
        max_p = max(s["prompt_ids"].shape[0] for s in samples)
        prompt_ids = torch.full((b, max_p), dec_pad_id, dtype=torch.long)
        prompt_mask = torch.zeros(b, max_p, dtype=torch.long)

        # Full prompt (doc + question) for the baseline_llm_full oracle.
        has_full = "full_prompt_ids" in samples[0]
        if has_full:
            max_fp = max(s["full_prompt_ids"].shape[0] for s in samples)
            full_prompt_ids = torch.full((b, max_fp), dec_pad_id, dtype=torch.long)
            full_prompt_mask = torch.zeros(b, max_fp, dtype=torch.long)

        # Answer: right-pad (standard).
        max_a = max(s["answer_ids"].shape[0] for s in samples)
        answer_ids = torch.full((b, max_a), dec_pad_id, dtype=torch.long)
        answer_labels = torch.full((b, max_a), -100, dtype=torch.long)
        answer_mask = torch.zeros(b, max_a, dtype=torch.long)

        # MIXED-FIDELITY extras: decoder-tokenised document (right-pad with the
        # decoder pad id; validity is bounded by the spans, no separate mask
        # needed) + per-chunk decoder spans (pad with (0, 0); chunk_counts says
        # how many rows are real per sample).
        has_mixed = "doc_dec_ids" in samples[0]
        if has_mixed:
            max_d = max(max(s["doc_dec_ids"].shape[0] for s in samples), 1)
            doc_dec_ids = torch.full((b, max_d), dec_pad_id, dtype=torch.long)
            max_c = max(max(s["chunk_dec_spans"].shape[0] for s in samples), 1)
            chunk_dec_spans = torch.zeros(b, max_c, 2, dtype=torch.long)
            chunk_counts = torch.zeros(b, dtype=torch.long)

        for i, s in enumerate(samples):
            sids, pids, aids = s["source_ids"], s["prompt_ids"], s["answer_ids"]
            sl, pl, al = sids.shape[0], pids.shape[0], aids.shape[0]
            source_ids[i, :sl] = sids
            source_attention_mask[i, :sl] = 1
            prompt_ids[i, max_p - pl:] = pids
            prompt_mask[i, max_p - pl:] = 1
            answer_ids[i, :al] = aids
            answer_labels[i, :al] = aids
            answer_mask[i, :al] = 1
            if has_full:
                fpids = s["full_prompt_ids"]
                fpl = fpids.shape[0]
                full_prompt_ids[i, max_fp - fpl:] = fpids
                full_prompt_mask[i, max_fp - fpl:] = 1
            if has_mixed:
                dids = s["doc_dec_ids"]
                doc_dec_ids[i, :dids.shape[0]] = dids
                spans = s["chunk_dec_spans"]
                chunk_dec_spans[i, :spans.shape[0]] = spans
                chunk_counts[i] = spans.shape[0]

        out = {
            "source_ids": source_ids,
            "source_attention_mask": source_attention_mask,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "answer_ids": answer_ids,
            "answer_labels": answer_labels,
            "answer_mask": answer_mask,
        }
        if has_full:
            out["full_prompt_ids"] = full_prompt_ids
            out["full_prompt_mask"] = full_prompt_mask
        if has_mixed:
            out["doc_dec_ids"] = doc_dec_ids
            out["chunk_dec_spans"] = chunk_dec_spans
            out["chunk_counts"] = chunk_counts
        return out

    return collate_cvlm_batch

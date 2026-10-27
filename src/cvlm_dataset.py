# Dataset + collation for CVLM.
#
# The on-the-fly variant: the dataset yields raw token ids for (source, prompt,
# answer). The model's frozen text encoder embeds `source_ids` at forward time,
# so there is no precomputed embeddings cache anywhere.

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


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
        source_col: str = 'input',
        prompt_col: str = 'prompt',
        answer_col: str = 'answer',
        task_col: str | None = None,
        prompt_template: str | None = None,
        multi_answer: bool = False,
        disable_length_filter: bool = False,
        hf_config: str | None = None,
        truncate_len: int = 0,
        use_chat_template: bool = False,
    ) -> None:
        # Column mapping decouples this dataset from PwC's exact schema so the
        # same class can serve LongBench eval (context/input/answers/dataset).
        #   source_col : document to be compressed into vision tokens.
        #   prompt_col : the user question (stays uncompressed).
        #   answer_col : gold answer. If ``multi_answer`` it is a List[str].
        #   task_col   : per-row task/subtask name (LongBench "dataset" field).
        # ``prompt_template`` is the official LongBench template with {context}
        # and {input} placeholders. When set, the compressed document replaces
        # {context} (dropped from the decoder prompt) and the remaining wrapper
        # text — instruction prefix + "Question: {input}\nAnswer:" — becomes the
        # uncompressed decoder prompt. ``disable_length_filter`` keeps every
        # eval sample (we raise max_source_len instead of dropping samples).
        self.source_col = source_col
        self.prompt_col = prompt_col
        self.answer_col = answer_col
        self.task_col = task_col
        self.prompt_template = prompt_template
        self.multi_answer = multi_answer
        # >0 enables the truncation baseline field: the decoder sees the LAST
        # ``truncate_len`` decoder-tokens of the context (uncompressed), with
        # earlier context discarded. Matches the "truncation baseline" from the
        # compression literature.
        self.truncate_len = int(truncate_len)
        # When True, wrap decoder text in the decoder's chat template (system +
        # user + assistant turns) for train/eval consistency with the instruct
        # model. The compressed document occupies its slot inside the user turn,
        # so the templated wrapper is split around the document: everything
        # before it becomes ``prompt_ids`` (seen before the vision block) and
        # everything after it — document suffix + question + assistant-open +
        # answer — becomes ``answer_ids``, with only the gold answer scored.
        self.use_chat_template = bool(use_chat_template)
        if hf_config is not None:
            self._hf: HFDataset = load_dataset(hf_dataset_name, hf_config, split=hf_split)
        else:
            self._hf = load_dataset(hf_dataset_name, split=hf_split)
        if max_samples > 0:
            self._hf = self._hf.select(range(min(max_samples, len(self._hf))))
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len
        self.max_source_len = max_source_len

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

        # LongBench eval: keep every sample. We size max_source_len large enough
        # to hold the longest document rather than dropping any evaluation row.
        if disable_length_filter:
            self._row_indices: List[int] = list(range(len(self._hf)))
            print(f'CvlmTrainDataset: eval mode, kept all {len(self._row_indices)} samples')
            return

        # Vectorised length pass: one batched fast-tokenizer call per field.
        inputs = [str(x) for x in self._hf[self.source_col]]
        prompts = [str(x) for x in self._hf[self.prompt_col]]
        answers = [self._first_answer(a) for a in self._hf[self.answer_col]]

        print(f'Tokenising {len(inputs)} samples to compute length filter...')
        enc_lens = self._batched_lengths(self._enc_tok, inputs)  # source lens in encoder tokens
        dec_prompt_lens = self._batched_lengths(
            self._dec_tok, prompts
        )  # question lens in decoder tokens
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
        self._row_indices = keep
        print(
            f'CvlmTrainDataset: kept {len(keep)}/{len(inputs)} samples '
            f'(source_tokens<={max_source_len}, prompt_tokens<={max_prompt_len}, '
            f'answer_tokens<={max_answer_len})'
        )
        if not keep:
            raise RuntimeError('No samples passed length filters; loosen caps.')

    @staticmethod
    def _first_answer(a: Any) -> str:
        """Ground truth for the length pass: first element if it is a list."""
        if isinstance(a, (list, tuple)):
            return str(a[0]) if a else ''
        return str(a)

    @staticmethod
    def _batched_lengths(
        tokenizer: PreTrainedTokenizer, texts: List[str], batch: int = 1024
    ) -> List[int]:
        out: List[int] = []
        for start in range(0, len(texts), batch):
            enc = tokenizer(
                [str(t) for t in texts[start : start + batch]],
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            out.extend(len(ids) for ids in enc['input_ids'])
        return out

    # Sentinel marking the compressed document's position inside the user turn.
    # Picked to be a single string that never collides with real content; we
    # split the rendered chat template on it to recover the pre-/post-document
    # text spans, then tokenise each span separately.
    _DOC_SENTINEL = '\ue000CVLM_DOC\ue000'

    def _apply_template(self, user_content: str, answer_text: str) -> str:
        """Render system+user+assistant turns with the decoder chat template.

        ``add_generation_prompt`` is implied by appending the assistant turn.
        The default system prompt is kept (protocol consistency with the
        instruct model). Returns the full templated string including the gold
        answer and its closing token.
        """
        msgs = [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': answer_text},
        ]
        return self._dec_tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

    def _template_gen_prompt(self, user_content: str) -> str:
        """Templated wrapper ending at the assistant-open tag (for generation)."""
        msgs = [{'role': 'user', 'content': user_content}]
        return self._dec_tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def _split_templated(self, user_content_with_doc: str, answer_text: str):
        """Split the templated sequence at the compressed-document boundary.

        ``user_content_with_doc`` is the user-turn content with ``_DOC_SENTINEL``
        marking where the compressed document sits. Returns
        (prompt_ids, answer_ids, answer_prompt_len, gen_suffix_ids):
          prompt_ids       : tokens BEFORE the document (system + user-open + any
                             user text preceding the document).
          answer_ids       : tokens AFTER the document — document suffix +
                             question + assistant-open + gold answer + closing.
          answer_prompt_len: #tokens in answer_ids that precede the gold answer
                             (masked to -100 so only the answer is scored).
          gen_suffix_ids   : the answer-prefix tokens (document suffix + question
                             + assistant-open) — fed after the vision block at
                             generation so decoding starts on the answer.
        """
        full = self._apply_template(user_content_with_doc, answer_text)
        assert self._DOC_SENTINEL in full, 'document sentinel lost in template'
        pre_doc, post_doc_full = full.split(self._DOC_SENTINEL, 1)
        # post_doc_full = <doc-suffix + question><|im_end|>\n<|im_start|>assistant\n<answer><|im_end|>\n
        gen_prompt = self._template_gen_prompt(user_content_with_doc)
        _, post_doc_prefix = gen_prompt.split(self._DOC_SENTINEL, 1)
        assert post_doc_full.startswith(post_doc_prefix), 'template prefix mismatch'

        # ``pre_doc`` is separated from everything after it by the vision block,
        # so tokenising it alone introduces no cross-boundary merge (vision, not
        # text, follows it). Everything after the document is contiguous text, so
        # it MUST be tokenised jointly to match real decoder tokenisation — then
        # the answer boundary is recovered by token-prefix length. Tokenising the
        # prefix and the tail separately would drift (control-token/newline
        # merges), mis-aligning the label mask.
        prompt_ids = self._dec_tok(pre_doc, add_special_tokens=False)['input_ids']
        answer_ids = self._dec_tok(post_doc_full, add_special_tokens=False)['input_ids']
        prefix_ids = self._dec_tok(post_doc_prefix, add_special_tokens=False)['input_ids']
        # The gen prefix is a proper token-prefix of the jointly-tokenised answer
        # because the assistant-open tag ends in "\n", a clean token boundary the
        # answer text does not merge across.
        answer_prompt_len = len(prefix_ids)
        assert answer_ids[:answer_prompt_len] == prefix_ids, (
            'answer prefix is not a clean token-prefix of the joint answer; '
            'template split would misalign the label mask'
        )
        return prompt_ids, answer_ids, answer_prompt_len, prefix_ids

    def __len__(self) -> int:
        return len(self._row_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_id = self._row_indices[idx]
        record = self._hf[row_id]
        source_text = str(record[self.source_col])
        question_text = str(record[self.prompt_col])

        src = self._enc_tok(
            source_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_source_len,
            return_tensors=None,
        )

        # Answers: raw text passthrough for reference-based scoring (LongBench is
        # multi-answer). answer_ids/labels still come from the FIRST answer so the
        # training / NLL path is unchanged.
        raw = record[self.answer_col]
        if self.multi_answer:
            answers_text = [str(x) for x in raw] if isinstance(raw, (list, tuple)) else [str(raw)]
        else:
            answers_text = [str(raw)]
        answer_text = answers_text[0]

        # Build the user-turn content with a sentinel where the compressed
        # document goes, and the fully-inlined version (document inlined) used by
        # the oracle / truncation prompts.
        if self.prompt_template is not None:
            # LongBench: official template with {context} (compressed) + {input}.
            user_with_doc = self.prompt_template.format(
                context=self._DOC_SENTINEL, input=question_text
            )
            user_inline = self.prompt_template.format(
                context=source_text, input=question_text
            )
        else:
            # PwC: user turn = document + blank line + question.
            user_with_doc = self._DOC_SENTINEL + '\n\n' + question_text
            user_inline = source_text + '\n\n' + question_text

        gen_suffix_ids = None
        answer_prompt_len = 0
        if self.use_chat_template:
            # Chat-templated: split the rendered template at the document so the
            # compressed document (vision) sits between prompt_ids and answer_ids.
            prompt_ids, answer_ids, answer_prompt_len, gen_suffix_ids = self._split_templated(
                user_with_doc, answer_text
            )
            # Oracle / truncation prompts are generation-only: full templated
            # wrapper (document inlined) ending at the assistant-open tag.
            full_prompt_ids = self._dec_tok(
                self._template_gen_prompt(user_inline), add_special_tokens=False
            )['input_ids']
        else:
            # Raw-text (legacy) path: prompt = question only; oracle = doc+question.
            if self.prompt_template is not None:
                _, _, suffix = self.prompt_template.partition('{context}')
                decoder_prompt_text = suffix.replace('{input}', question_text)
            else:
                decoder_prompt_text = question_text
            prompt_ids = self._dec_tok(decoder_prompt_text, add_special_tokens=False)['input_ids']
            answer_ids = self._dec_tok(answer_text, add_special_tokens=False)['input_ids']
            full_prompt_ids = self._dec_tok(user_inline, add_special_tokens=False)['input_ids']

        # Truncation baseline: keep the LAST ``truncate_len`` decoder-tokens of
        # the context, discard the earlier context, then re-assemble with the
        # (untruncated) wrapper + question. Only the document is shortened, so the
        # decoder sees a well-formed prompt over the tail of the context.
        trunc_ids = None
        if self.truncate_len > 0:
            ctx_ids = self._dec_tok(
                source_text, add_special_tokens=False, truncation=False
            )['input_ids']
            kept_ctx = self._dec_tok.decode(ctx_ids[-self.truncate_len:])
            if self.prompt_template is not None:
                trunc_user = self.prompt_template.format(context=kept_ctx, input=question_text)
            else:
                trunc_user = kept_ctx + '\n\n' + question_text
            if self.use_chat_template:
                trunc_ids = self._dec_tok(
                    self._template_gen_prompt(trunc_user), add_special_tokens=False
                )['input_ids']
            else:
                trunc_ids = self._dec_tok(trunc_user, add_special_tokens=False)['input_ids']

        task_name = str(record[self.task_col]) if self.task_col is not None else ''

        out = {
            'source_ids': torch.as_tensor(src['input_ids'], dtype=torch.long),
            'prompt_ids': torch.as_tensor(prompt_ids, dtype=torch.long),
            'full_prompt_ids': torch.as_tensor(full_prompt_ids, dtype=torch.long),
            'answer_ids': torch.as_tensor(answer_ids, dtype=torch.long),
            # #answer tokens preceding the gold answer (chat-template prefix:
            # document-suffix + question + assistant-open). Masked to -100 in the
            # collator so only the gold answer is scored. 0 in the raw path.
            'answer_prompt_len': int(answer_prompt_len),
            'answers_text': answers_text,
            'task_name': task_name,
        }
        # Chat-template generation suffix (document-suffix + question +
        # assistant-open): fed after the vision block at generation so decoding
        # begins on the answer. Absent in the raw path.
        if gen_suffix_ids is not None:
            out['gen_suffix_ids'] = torch.as_tensor(gen_suffix_ids, dtype=torch.long)
        if trunc_ids is not None:
            out['trunc_prompt_ids'] = torch.as_tensor(trunc_ids, dtype=torch.long)
        return out


def make_collate_fn(
    dec_pad_id: int,
    enc_pad_id: int,
) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    def collate_cvlm_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        b = len(samples)

        # Source (encoder input): right-pad with encoder pad id.
        max_s = max(s['source_ids'].shape[0] for s in samples)
        source_ids = torch.full((b, max_s), enc_pad_id, dtype=torch.long)
        source_attention_mask = torch.zeros(b, max_s, dtype=torch.long)

        # Prompt: LEFT-pad so real tokens sit flush against the vision block.
        max_p = max(s['prompt_ids'].shape[0] for s in samples)
        prompt_ids = torch.full((b, max_p), dec_pad_id, dtype=torch.long)
        prompt_mask = torch.zeros(b, max_p, dtype=torch.long)

        # Full prompt (doc + question) for the baseline_llm_full oracle.
        has_full = 'full_prompt_ids' in samples[0]
        if has_full:
            max_fp = max(s['full_prompt_ids'].shape[0] for s in samples)
            full_prompt_ids = torch.full((b, max_fp), dec_pad_id, dtype=torch.long)
            full_prompt_mask = torch.zeros(b, max_fp, dtype=torch.long)

        # Truncated prompt (last-n context + question) for baseline_truncate.
        # LEFT-pad like the other decoder-generation prompts.
        has_trunc = 'trunc_prompt_ids' in samples[0]
        if has_trunc:
            max_tp = max(s['trunc_prompt_ids'].shape[0] for s in samples)
            trunc_prompt_ids = torch.full((b, max_tp), dec_pad_id, dtype=torch.long)
            trunc_prompt_mask = torch.zeros(b, max_tp, dtype=torch.long)

        # Chat-template generation suffix (question + assistant-open) fed after
        # the vision block at generation. LEFT-pad so real tokens sit flush after
        # vision. Empty samples (len 0) are allowed and yield an all-pad row.
        has_suffix = 'gen_suffix_ids' in samples[0]
        if has_suffix:
            max_gs = max(s['gen_suffix_ids'].shape[0] for s in samples)
            gen_suffix_ids = torch.full((b, max_gs), dec_pad_id, dtype=torch.long)
            gen_suffix_mask = torch.zeros(b, max_gs, dtype=torch.long)

        # Answer: right-pad (standard).
        max_a = max(s['answer_ids'].shape[0] for s in samples)
        answer_ids = torch.full((b, max_a), dec_pad_id, dtype=torch.long)
        answer_labels = torch.full((b, max_a), -100, dtype=torch.long)
        answer_mask = torch.zeros(b, max_a, dtype=torch.long)

        for i, s in enumerate(samples):
            sids, pids, aids = s['source_ids'], s['prompt_ids'], s['answer_ids']
            sl, pl, al = sids.shape[0], pids.shape[0], aids.shape[0]
            source_ids[i, :sl] = sids
            source_attention_mask[i, :sl] = 1
            prompt_ids[i, max_p - pl :] = pids
            prompt_mask[i, max_p - pl :] = 1
            answer_ids[i, :al] = aids
            answer_labels[i, :al] = aids
            answer_mask[i, :al] = 1
            # Chat-template: mask the answer-prefix (document-suffix + question +
            # assistant-open) so loss is scored only on the gold answer. The
            # decoder still attends over these tokens (answer_mask stays 1); only
            # their labels are ignored. answer_prompt_len is 0 in the raw path.
            apl = int(s.get('answer_prompt_len', 0))
            if apl > 0:
                answer_labels[i, :apl] = -100
            if has_suffix:
                gsids = s['gen_suffix_ids']
                gsl = gsids.shape[0]
                if gsl > 0:
                    gen_suffix_ids[i, max_gs - gsl :] = gsids
                    gen_suffix_mask[i, max_gs - gsl :] = 1
            if has_full:
                fpids = s['full_prompt_ids']
                fpl = fpids.shape[0]
                full_prompt_ids[i, max_fp - fpl :] = fpids
                full_prompt_mask[i, max_fp - fpl :] = 1
            if has_trunc:
                tpids = s['trunc_prompt_ids']
                tpl = tpids.shape[0]
                trunc_prompt_ids[i, max_tp - tpl :] = tpids
                trunc_prompt_mask[i, max_tp - tpl :] = 1

        out = {
            'source_ids': source_ids,
            'source_attention_mask': source_attention_mask,
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'answer_ids': answer_ids,
            'answer_labels': answer_labels,
            'answer_mask': answer_mask,
        }
        if has_full:
            out['full_prompt_ids'] = full_prompt_ids
            out['full_prompt_mask'] = full_prompt_mask
        if has_trunc:
            out['trunc_prompt_ids'] = trunc_prompt_ids
            out['trunc_prompt_mask'] = trunc_prompt_mask
        if has_suffix:
            out['gen_suffix_ids'] = gen_suffix_ids
            out['gen_suffix_mask'] = gen_suffix_mask
        # Non-tensor passthrough for reference-based scoring (LongBench).
        if 'answers_text' in samples[0]:
            out['answers_text'] = [s['answers_text'] for s in samples]
        if 'task_name' in samples[0]:
            out['task_name'] = [s['task_name'] for s in samples]
        return out

    return collate_cvlm_batch

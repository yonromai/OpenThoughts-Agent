"""Lever-2 summarization-trace -> SFT format utilities.

Standalone post-processor for the rows emitted by harbor's
`collect_conversations_from_trial` for summarizing terminus-2 traces. Validated
2026-06-17 on real t16384-32k pilot traces: NAIVE rows are dropped by LF;
fixed rows round-trip through the Qwen3-8B `qwen3` template with the terminate
(`task_complete: true`) + `<|im_end|>` EOS confirmed loss-bearing. See
experiments/ablation_exploration_in_rl/plan_lever2_summarization_aware_sft.md
("SFT formatting utilities") for the full design + prototype results.

Why it exists (validated, see proto report):
  The harbor exporter emits the post-summarization continuation episode with the
  task/system prompt as a leading role:"user" turn, immediately followed by the
  summarization handoff (also role:"user"). The result is TWO consecutive leading
  user turns. LLaMA-Factory's SharegptDatasetConverter enforces strict
  user/assistant alternation (accept_tags[turn_idx % 2]); two leading user turns
  trip `broken_data=True` on turn_idx==1 and the row is SILENTLY DROPPED.

Public entry point: fix_summarization_row(conv) -> conv (mutated copy).
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List

HANDOFF_SIGNATURES = (
    "You are picking up work from a previous AI agent",
    "Here are the answers the other agent provided",
    "Continue working on this task from where the previous agent left off",
    "You can no longer ask questions",
)


def _is_handoff(text: str) -> bool:
    return any(sig in (text or "") for sig in HANDOFF_SIGNATURES)


def fix_summarization_row(conv: Dict[str, Any]) -> Dict[str, Any]:
    """Make a harbor 'last'/'all'-episode row LF-loadable.

    Transformations (idempotent, order matters):
      1. PROMOTE leading system: if the row starts with >=2 consecutive user
         turns AND the 2nd is a handoff (or any 2 consecutive leading users),
         relabel msgs[0] (the task/system prompt) role 'user' -> 'system'.
         LF then extracts it via system_tag and the remaining sequence
         alternates user/assistant cleanly.
      2. DEDUPE is_copied_context echo: collapse any *adjacent* duplicate-role
         turns that remain after step 1 by merging their content with a
         newline (handles the rare double-handoff / copied-context echo).
      3. Leave assistant turns (incl. the terminate `task_complete: true` text
         + EOS) untouched so they stay loss-bearing.

    Returns a deep-copied, mutated conv. Does not touch already-correct rows.
    """
    out = copy.deepcopy(conv)
    msgs: List[Dict[str, str]] = out.get("conversations", [])
    if not msgs:
        return out

    # --- 1. promote leading system ---
    if (
        len(msgs) >= 2
        and msgs[0].get("role") == "user"
        and msgs[1].get("role") == "user"
    ):
        # The first user turn is the original task/system prompt; the second is
        # the summarization handoff. Promote the first to system.
        msgs[0] = {"role": "system", "content": msgs[0]["content"]}

    # --- 2. dedupe adjacent same-role turns (post-promotion) ---
    merged: List[Dict[str, str]] = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
            # adjacent duplicate (user/user or assistant/assistant) -> merge.
            # Skip exact-duplicate copied-context echoes.
            if m["content"].strip() and m["content"].strip() not in merged[-1]["content"]:
                merged[-1] = {
                    "role": m["role"],
                    "content": merged[-1]["content"] + "\n" + m["content"],
                }
            # else: pure echo, drop it
        else:
            merged.append(dict(m))
    out["conversations"] = merged
    return out


def validate_lf_loadable(conv: Dict[str, Any]) -> Dict[str, Any]:
    """Round-trip a row through LF SharegptDatasetConverter + qwen3 template.

    Returns a dict with: loadable(bool), n_messages, ends_on_assistant,
    terminate_in_last_assistant, dropped_by_lf(bool).
    """
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr

    attr = DatasetAttr(load_from="file", dataset_name="proto")
    # ShareGPT role/content tags (the locked masking requirement)
    attr.messages = "conversations"
    attr.role_tag = "role"
    attr.content_tag = "content"
    attr.user_tag = "user"
    attr.assistant_tag = "assistant"
    attr.system_tag = "system"
    attr.observation_tag = "observation"
    attr.function_tag = "function_call"

    conv_fn = SharegptDatasetConverter(dataset_attr=attr, data_args=None)
    example = {"conversations": conv["conversations"]}
    result = conv_fn(example)
    prompt, response = result["_prompt"], result["_response"]
    dropped = len(prompt) == 0 and len(response) == 0
    last_assist_txt = response[-1]["content"] if response else ""
    return {
        "loadable": not dropped,
        "dropped_by_lf": dropped,
        "n_prompt": len(prompt),
        "n_response": len(response),
        "ends_on_assistant": bool(response),
        "terminate_in_last_assistant": '"task_complete": true' in last_assist_txt,
    }

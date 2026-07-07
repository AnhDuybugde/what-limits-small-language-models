import json
from pathlib import Path


def load_experiment_namespace():
    namespace = {"__name__": "__main__"}
    notebook = json.loads(Path("02-run-experiment.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"][1:7]:
        source = "".join(cell["source"])
        exec(compile(source, "02-run-experiment.ipynb", "exec"), namespace)
    return namespace


def main():
    ns = load_experiment_namespace()
    knowledge_sample = {
        "sample_id": "sfq_178",
        "axis": "knowledge",
        "dataset": "triviaqa",
        "question": "What general name is given to a rotating star which emits a regular beat of radiation?",
        "context": "",
        "gold_answer": "PULSAR",
        "supporting_facts": [],
        "metadata": {"source": "huggingface"},
    }
    reasoning_sample = {
        "sample_id": "gsm8k_test_000",
        "axis": "reasoning",
        "dataset": "gsm8k",
        "question": "A shop sells 3 packs with 4 pens each. How many pens are there?",
        "context": "",
        "gold_answer": "12",
        "supporting_facts": [],
        "metadata": {"source": "huggingface"},
    }
    context_sample = {
        "sample_id": "hotpotqa_test_000",
        "axis": "context",
        "dataset": "hotpotqa",
        "question": "Which city hosted the example games?",
        "context": "Gold: The example games were hosted in Paris.\n\nDistractor: London hosted another event.",
        "gold_answer": "Paris",
        "supporting_facts": {"title": ["Gold"], "sent_id": [0]},
        "metadata": {"source": "huggingface"},
    }

    prompt = ns["build_prompt"](knowledge_sample, [knowledge_sample])
    if "Final answer:" not in prompt:
        raise AssertionError(prompt)

    condition_samples = {
        "reasoning_direct": ("reasoning", reasoning_sample),
        "reasoning_cot": ("reasoning", reasoning_sample),
        "reasoning_oracle_step": ("reasoning", reasoning_sample),
        "context_none": ("context", context_sample),
        "context_retrieved": ("context", context_sample),
        "context_oracle": ("context", context_sample),
        "context_oracle_distractor": ("context", context_sample),
        "context_oracle_end": ("context", context_sample),
        "robust_original": ("robustness", reasoning_sample),
        "robust_para": ("robustness", reasoning_sample),
        "robust_typo": ("robustness", reasoning_sample),
        "robust_format": ("robustness", context_sample),
    }
    for condition, (axis, sample) in condition_samples.items():
        ns["AXIS"] = axis
        ns["CONDITION"] = condition
        prompt = ns["build_prompt"](sample, [sample])
        if condition == "robust_format":
            if '{"answer": "..."}' not in prompt:
                raise AssertionError(prompt)
        elif "Final answer:" not in prompt:
            raise AssertionError(prompt)

    ns["AXIS"] = "robustness"
    ns["CONDITION"] = "robust_original"
    original_prompt = ns["build_prompt"](reasoning_sample, [reasoning_sample])
    ns["CONDITION"] = "robust_para"
    paraphrase_prompt = ns["build_prompt"](reasoning_sample, [reasoning_sample])
    ns["CONDITION"] = "robust_typo"
    typo_prompt = ns["build_prompt"](reasoning_sample, [reasoning_sample])
    ns["CONDITION"] = "robust_format"
    format_prompt = ns["build_prompt"](reasoning_sample, [reasoning_sample])
    if "Problem:" not in original_prompt or "Problem:" not in paraphrase_prompt:
        raise AssertionError((original_prompt, paraphrase_prompt))
    if "Final answer:" not in original_prompt or "Final answer:" not in paraphrase_prompt:
        raise AssertionError((original_prompt, paraphrase_prompt))
    if typo_prompt == original_prompt:
        raise AssertionError("robust_typo must perturb the GSM8K problem text.")
    if reasoning_sample["question"] in typo_prompt:
        raise AssertionError(typo_prompt)
    if "Return only one number" not in typo_prompt:
        raise AssertionError(typo_prompt)
    if '{"answer": "<number>"}' not in format_prompt:
        raise AssertionError(format_prompt)
    forbidden_reasoning_cues = [
        "step by step",
        "think step",
        "reason carefully",
        "show your work",
        "explain your reasoning",
        "let's think",
    ]
    lowered_para = paraphrase_prompt.lower()
    if any(cue in lowered_para for cue in forbidden_reasoning_cues):
        raise AssertionError(paraphrase_prompt)

    ns["AXIS"] = "context"
    expected_context = {
        "context_none": ("none", "none", 0, 0),
        "context_retrieved": ("bm25", "beginning", 2, 1),
        "context_oracle": ("oracle", "beginning", 1, 0),
        "context_oracle_distractor": ("oracle+distractor", "beginning", 2, 1),
        "context_oracle_end": ("oracle_end", "end", 2, 1),
    }
    for condition, (source, position, paragraphs, distractors) in expected_context.items():
        ns["CONDITION"] = condition
        metadata = ns["record_context_metadata"](context_sample)
        if metadata["context_source"] != source:
            raise AssertionError((condition, metadata))
        if metadata["gold_at_position"] != position:
            raise AssertionError((condition, metadata))
        if metadata["num_context_paragraphs"] != paragraphs:
            raise AssertionError((condition, metadata))
        if metadata["num_distractor_paragraphs"] != distractors:
            raise AssertionError((condition, metadata))
        if metadata["distractor_contains_answer"]:
            raise AssertionError((condition, metadata))

    if ns["extract_answer"]("Final answer: pulsar\nIt rotates.", "knowledge", "knowledge_oracle") != "pulsar":
        raise AssertionError("QA extraction should keep only the short answer line.")
    raw = "Final answer: John Entwhistle. To determine who played bass, I used the reference."
    if ns["extract_answer"](raw, "knowledge", "knowledge_oracle") != "John Entwhistle":
        raise AssertionError("QA extraction should remove explanation after the short answer.")
    if ns["extract_answer"]("We compute 3 + 4 = 7.", "reasoning", "reasoning_direct") != "7":
        raise AssertionError("Reasoning extraction should recover the final number.")
    exact_match, f1, accuracy = ns["score_prediction"]("5760 coins", "5760", reasoning_sample)
    if (exact_match, f1, accuracy) != (1.0, 1.0, 1.0):
        raise AssertionError((exact_match, f1, accuracy))
    if ns["extract_answer"]("Final answer: 5760 coins", "robustness", "robust_format") != "5760":
        raise AssertionError("Robust GSM8K extraction should recover numeric answers from non-JSON text.")
    if ns["extract_answer"]('{"answer": "Paris"}', "robustness", "robust_format") != "Paris":
        raise AssertionError("Robust JSON extraction should parse the answer field.")
    if not str(ns["OUTPUT_PATH"]).startswith("outputs\\records_experiment_robustness_"):
        raise AssertionError(ns["OUTPUT_PATH"])
    print("experiment runner checks passed")


if __name__ == "__main__":
    main()

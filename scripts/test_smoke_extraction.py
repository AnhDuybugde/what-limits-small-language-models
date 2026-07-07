import json
from pathlib import Path


def load_smoke_namespace():
    namespace = {"__name__": "__main__"}
    notebook = json.loads(Path("01-smoke-test.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"][1:4]:
        source = "".join(cell["source"])
        exec(compile(source, "01-smoke-test.ipynb", "exec"), namespace)
    return namespace


def main():
    ns = load_smoke_namespace()
    extract_answer = ns["extract_answer"]
    cases = [
        ("We compute 12 + 5 = 17.", "reasoning", "reasoning_direct", "17"),
        ("Step one gives 3. Final answer: 42", "reasoning", "reasoning_cot", "42"),
        ("The model rambles.\nFinal answer:\nParis", "knowledge", "knowledge_closed", "Paris"),
        ("Use context. Final answer: Alexander Fleming", "context", "context_oracle", "Alexander Fleming"),
        ("Final answer: pulsar\nIt is a rotating neutron star.", "knowledge", "knowledge_oracle", "pulsar"),
        ("The answer is: PULSAR\nThis follows from the passage.", "context", "context_oracle", "PULSAR"),
    ]
    for raw, axis, condition, expected in cases:
        actual = extract_answer(raw, axis, condition)
        if actual != expected:
            raise AssertionError((raw, actual, expected))
    print("extraction checks passed")


if __name__ == "__main__":
    main()

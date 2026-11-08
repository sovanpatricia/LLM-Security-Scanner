# Pilot case annotation guide — schema version 1

Each `cases/<case-folder>/case.json` describes one controlled case. These are
application-derived development fixtures, not an independently held-out benchmark.
Retain upstream attribution/license text and document every adaptation.

## Fields

| Field | Meaning and allowed values |
| --- | --- |
| `schema_version` | Integer `1`. |
| `case_id` | Unique identifier for this source case, shared by its variants. |
| `group_id` | Application/repository family used to keep related cases in one split. |
| `source.url`, `source.commit` | Original repository and exact upstream commit. |
| `source.license`, `source.license_file` | License identifier and a file path relative to the case folder. |
| `source.app_kind` | `intentionally_vulnerable_lab` or `real_world_app`. |
| `label.owasp`, `label.cwe` | Proposed OWASP LLM category and corresponding CWE; taxonomy mappings do not prove a label. |
| `variants.vulnerable`, `variants.control` | Each has `file` (case-relative), `origin`, and an evidence-based `description`. |
| `origin` | `original`: upstream target behavior retained; `mutated`: intentionally introduced weakness; `repaired`: protection added to upstream-derived code. Extraction/setup adaptations still belong in descriptions. |
| `excerpts` | Upstream locations: `role`, `file`, inclusive one-based `start_line`/`end_line`, and `symbol`. Exactly one `target`; all other excerpts are `context`. These paths refer to upstream source, not necessarily local files. |
| `context.method` | `auto_same_file`, `manual_same_file` or `manual_cross_file`. `manual_same_file` is for context in the same file that the automatic extractor cannot take, e.g. methods of a very large class. |
| `context` for `auto_same_file` | Include `extractor: "same-file-name-closure-v1"`, `root_symbol`, and `artifact`. Generated artifact paths are repo-relative; `{variant}` expands to `control` or `vulnerable`. |
| `context` for `manual_same_file` or `manual_cross_file` | Include `artifact`, relative to the case folder (currently `context.json`). Extra provenance fields may preserve previous annotations. |
| `data_flow.source`, `data_flow.sink` | Plain-language trust/data path, either model-controlled output to an operation or confidential data to the model. State simulations explicitly. |
| `validation.script`, `validation.report` | Existing files relative to the repository root. Run validation before the schema check to generate reports when needed. |
| `validation.live_llm`, `validation.summary` | Boolean indicating live inference and a description of what was actually demonstrated. Simulated dispatch does not prove live attack success. |
| `review.status`, `review.reviewers` | Review state and reviewer names/identifiers. Preserve existing states; an empty list means no reviewers are recorded. Use `uncertain` for unresolved labels. |
| `split`, `excluded_from_training` | Intended partition and boolean exclusion flag. Both current cases remain `pilot_development_only` and excluded. |
| `assumptions`, `limitations` | Lists of conditions required by the label and boundaries of the evidence. |
| `variant_of`, `form` | Optional. For a refactored variant of another case: the base `case_id` and the vulnerable form (e.g. `through_variable`, `through_helper`, `compile_eval`, `inline_exec`). A variant has the same `group_id` and must stay in the same split as its base. |
| `expected_observations` | Optional expected output details; retained for the sensitive-tool-result case. |

## Labeling rules

- A case is LLM05 only if the dangerous value actually comes from the LLM:
  a tool argument, completion text, or structured output. The same code with
  a value from configuration or a constant is not LLM05 merely because it
  contains a dangerous operation. Record source evidence; distinguish a
  simulated model output from a live model-generated value.
- A control is **protected against the tested attack**, never automatically SAFE.
- One target vulnerability per case. Cases with several vulnerabilities or
  unclear labels are marked uncertain or excluded. The first SQL case remains
  a diagnostic with unresolved authorization questions, not a clean SAFE pair.
- Code files must not reveal their label: no comments such as "repaired" or "vulnerable" and an identical header in both files. Explanations belong in case.json.
- Keep original/mutated/repaired variants, duplicates, and cases from the same
  application family together when partitioning data. Do not tune on test cases.
- Preserve annotation facts when restructuring metadata. An empty list states
  that nothing is recorded; do not invent reviewer names or review completion.

## Context extraction

`src/llmscan/context.py` parses without executing source. It collects a named
root and referenced same-file top-level declarations, retaining whole classes
and imports. External helpers are not resolved. Name shadowing may over-include
context; this is not data-flow analysis. Defaults: 20 declarations plus imports,
300 output lines. Exceeding a bound raises instead of silently truncating.
The extraction policy must be the same for training and evaluation.

## Checks (from the repository root)

```bash
python experiments/pilot/check_context_extraction.py
python experiments/pilot/check_first_case.py
python experiments/pilot/check_sensitive_case.py
python experiments/pilot/check_case_schema.py
```

The first SQL harness tests dummy SQLite data via an extracted wrapper and
simulated tool-call dispatch. The sensitive-result harness tests preserved
functionality, fixed-user behavior, and dummy-password exposure. Neither calls
a live LLM. The baseline model and scanner remain unchanged.

Reports use repository-relative forward-slash paths. Environment versions,
model warnings, and predictions may still differ across environments: portable
paths do not imply byte-identical experimental results.

The schema checker uses only the standard library. It checks required nested
fields, enumerations, exactly one target, valid line ordering, and existence of
variant/license/script/report files. It does not verify upstream line content,
scientific label correctness, or successful human review. New cases are found
automatically under `data/pilot/cases/*/case.json`.

## Split and dataset file (step 2)

`data/pilot/splits.json` assigns each application group to `app_validation` or
`app_test`. It was fixed before any training or evaluation on application
cases. A case's `split` field must equal its group's split; all variants and
both files of a case follow their group. Training uses the Lab 3 synthetic
data only.

`python data/build_appset.py` writes `data/appset.csv` with two rows per case:
`vulnerable.py` labelled with the case's OWASP category and `control.py`
labelled `SAFE` (protected against that case's target vulnerability). The
columns match the Lab 3 CSVs (`id, template, language, label, vulnerable,
code`) so the existing pipeline can read it with `pandas.read_csv`, plus
`case_id`, `pair_id`, `group`, `split`, `variant`, `origin`, `form`, `app_kind`
and `review_status`. `template` is the case family (base case and its
variants), so template-aware code keeps variants together.

`python experiments/pilot/check_appset.py` verifies that the CSV is current,
that no group, family or pair crosses splits, and that each split contains
LLM02, LLM05, LLM06 and SAFE.

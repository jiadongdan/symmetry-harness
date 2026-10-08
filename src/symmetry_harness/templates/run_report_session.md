# Fine-tuning Session Report

**Session:** `{session_id}` | **Completed fine-tuning runs:** {run_count}

_Report generated {generated_utc} by symmetry-harness {harness_version}._

## 1. Session at a Glance

This session contains {run_count} independent fine-tuning results. Each image
keeps its own annotations, fitted model, prediction and reproducibility record.
The runs are collected here for convenient review; they were not combined into
one joint training dataset.

{comparison_status}

{run_table}

## 2. Results by Image

{run_sections}

## 3. Analysis Context

{notes}

## 4. Interpretation Boundaries

- Class proportions describe model assignments on each prediction grid, not measured material fractions.
- Areas marked for review reflect model uncertainty, not confirmed errors.
- Support-point fit is not an independent estimate of performance on unseen regions.
- Compare runs quantitatively only when their class meanings and analysis settings are equivalent.

## 5. Reproducibility

{reproduction_table}

Each linked image report contains its complete settings, checksums, artifacts
and reproduction command.

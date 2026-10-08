# Symmetry Analysis Run Report

**Run:** `{run_id}` | **Technical status:** `{status}` | **Analysis type:** {kind}

_Report generated {generated_utc} by symmetry-harness {harness_version}._

## 1. Result at a Glance

{summary}

{evidence_status}

{class_distribution_table}

## 2. Main Visual Evidence

{fig_overview}

{result_readout}

## 3. Analysis Context

{notes}

## 4. Input and Support Annotations

{fig_input}

{fig_support}

{annotation_table}

{annotation_details}

## 5. Prediction Certainty and Fine-tuning

### Where the model is less certain

{uncertainty_table}

### Fit to the selected support points

{training_interpretation}

{fig_training}

<details>
<summary>Fine-tuning settings and numerical diagnostics</summary>

{training_table}

</details>

## 6. Symmetry Representation

The eight channels are a fixed representation: the source image plus one
reflection-strength map, two signed reflection-orientation maps, and one
signed rotation-response map per fold. Source intensity and reflection strength
use `[0, 1]`; all orientation and rotation-response channels use `[-1, 1]`.
These fixed ranges make maps comparable across runs without per-image
renormalization.

{fig_symmetry}

{feature_table}

## 7. Interpretation Boundaries

{caveats}

## 8. Reproducibility

{workflow}

{configuration_table}

Reproduce this run from its versioned record:

```text
symmetry reproduce --record "{record_path}"
```

## Appendix A. Artifacts

{artifact_list}

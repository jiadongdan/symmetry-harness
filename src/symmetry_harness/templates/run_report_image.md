# Symmetry Analysis Run Report

- **Run ID:** `{run_id}`
- **Kind:** {kind}
- **Status:** `{status}`
- **Generated:** {generated_utc} by `symmetry report` (symmetry-harness {harness_version})

## 1. Summary

{summary}

## 2. Workflow Overview

{workflow}

## 3. Session Notes

{notes}

## 4. Input & Annotation

{fig_input}

{fig_support}

{annotation_table}

## 5. Symmetry Features

The eight channels are a fixed representation: the source image plus one
reflection-strength map, two signed reflection-orientation maps, and one
rotation map per fold. Every tile is rendered against a fixed cross-image colour
range, so tiles are comparable across runs and never renormalized per image.

{fig_symmetry}

{feature_table}

## 6. Fine-tuning

{fig_training}

{training_table}

## 7. Dense Prediction

{fig_mask}

{fig_overlay}

{fig_confidence}

{fig_entropy}

{prediction_table}

## 8. Configuration & Provenance

{configuration_table}

Reproduce this run from its versioned record:

```text
symmetry reproduce --record "{record_path}"
```

## 9. Caveats

{caveats}

## 10. Artifacts

{artifact_list}

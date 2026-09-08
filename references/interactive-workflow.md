# Interactive Workflow

Read this reference only after the local interface is available and the user asks
for workflow help, analysis assistance, or result interpretation.

1. Let the user choose the installed model, weight, and local image in the
   interface. Do not upload the image to a remote service.
2. Ask the user to define the local classes and select at least three, preferably
   five, representative support points per class.
3. Do not choose support points on the user's behalf. The interface rejects points
   whose full classifier patch would cross the image boundary.
4. Ask the user to review class names, colors, counts, patch outlines, and patch
   previews before starting fine-tuning.
5. Let the interface run adapter-plus-head fine-tuning and dense prediction.
   Report the run directory, support counts, configuration, artifacts, and
   warnings.
6. If the map is not satisfactory, invite the user to add representative points
   and create a new run. Never overwrite an earlier run.

## Scientific Invariants

- Treat selected points as support data, not independent validation data.
- Never describe support accuracy or training loss as generalization accuracy.
- Keep the pretrained checkpoint read-only and save adapter-plus-head weights
  separately with the base checkpoint checksum.
- Record image checksum, coordinates, class mapping, feature parameters,
  classifier patch size, stride, seed, runtime, and software versions.
- Preserve exact source-image coordinates. Do not substitute display coordinates
  for array coordinates without validation.
- Confidence and entropy describe model output, not physical correctness.
- Keep unpublished images local unless the user explicitly authorizes upload.

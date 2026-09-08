# Headless Execution and Reproduction

Read this reference only when the user asks to execute a saved annotation session
or reproduce a completed run.

- Run a saved annotation session with:

  ```bash
  symmetry run --input <path> --annotations <session.json>
  ```

- Reproduce a completed run with:

  ```bash
  symmetry reproduce --record <run_record.json>
  ```

Use only the recorded scientific and training options. Report the output run
directory, generated artifacts, and warnings. Never overwrite an earlier run.

# Project Structure Guide (Concise)

## 1. Directory Roles

- **data/**: Original datasets & ground truth only.
- **docs/**: All documentation, specs, and guides.
- **src/**: Source code logic for each methodology (modular, reusable, no scripts).
- **experiments/**: User-facing interfaces for running experiments (prefer Jupyter notebooks; scripts only for archiving/utilities).
- **results/**: All experiment outputs, organized by methodology and timestamp.

---

## 2. Naming & Organization

- **Methodology/Feature**: Use `snake_case` for all new methods/features (e.g., `llm_pure`, `llm_postfilter`, `my_new_method`).
- **src/[method]/**: All core logic for a method/feature.
- **experiments/[method]/**: Notebooks for running, analyzing, and validating the method. Use scripts only for archiving, cleaning, or automation.
- **results/[method]/**: All outputs, with subfolders for each experiment run (timestamped).

---

## 3. Adding a New Method/Feature

1. **Document**:
   - Add a spec/guide in `docs/[method].md`.
2. **Code**:
   - Add logic in `src/[method]/`.
3. **Interface**:
   - Add a Jupyter notebook in `experiments/[method]/notebooks/` as the main user interface.
   - Use scripts in `experiments/[method]/` only for archiving, cleaning, or batch automation.
4. **Results**:
   - Store all outputs in `results/[method]/[timestamp_or_name]/`.

---

## 4. Best Practices

- **Prefer Notebooks**: For new features, use notebooks as the main interface for running and analyzing experiments.
- **Consistent Naming**: Keep directory and file names consistent across `src/`, `experiments/`, and `results/`.
- **Separation of Concerns**:
  - Logic in `src/`
  - User interface in `experiments/`
  - Outputs in `results/`
- **Archive & Clean**: Always archive results and clean working directories after each experiment.

---

## 5. Example Structure

```
src/
  llm_pure/
  llm_postfilter/
  my_new_method/
experiments/
  llm_pure/
  llm_postfilter/
  my_new_method/
    notebooks/
      01_intro.ipynb
      02_experiment.ipynb
    archive_results.py
    clean_working_dir.py
results/
  llm_pure/
  llm_postfilter/
  my_new_method/
docs/
  LLM_Pure.md
  LLM_PostFilter.md
  My_New_Method.md
```

---

**Summary:**

- Use notebooks as the main interface for new features.
- Keep logic, interface, and results strictly separated.
- Mirror naming and structure across all main directories.
- Archive and document every experiment.

This keeps the project clean, scalable, and easy for anyone to contribute or extend!

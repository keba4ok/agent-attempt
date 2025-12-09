# Results Repository Setup

This project generates results (logs, statistics) that should be stored in a separate repository.

## Submodule stuff

A git submodule allows you to have a separate repository inside your main repository.

### Setup

1. **Create the results repository on GitHub**

2. **Add it as a submodule**

3. **Initialize the submodule** 

### Using the Submodule

Results will be stored in `results_repo/` directory:
- `results_repo/results_dpaia/`
- `results_repo/results_swe/`
- `results_repo/statistics/`

### Pushing Results

Usa a script `push_results.sh`:

```bash
chmod +x push_results.sh
```

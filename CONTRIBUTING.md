# Contributing

Thanks for your interest! Bug reports, new tasks and new analysis charts are all welcome.

1. Fork the repository and create a branch from `main`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Keep changes focused; follow the existing style (type hints, short docstrings, no dead code).
4. Check that the analysis still runs on the bundled sample:
   `python -m llm_cost_per_success general analyze --experiment examples/sample-run`
5. Open a pull request describing what changed and why.

Never commit API keys or the `.env` file.

## Adding tasks

Add an entry to `data/tasks.json` with a unique `id`, a `category` key from `categories`,
the `prompt` and the `expected` answer. The task fingerprint stored in each experiment
manifest changes automatically, so older results stay comparable by their hash.

# Development

Python 3.12 and Node 22 are required. Install with `pip install -e '.[dev]'` and, in `web`,
`npm install`. Run `ruff check .`, `mypy src`, `pytest`, `npm test`, and `npm run e2e`. Keep database,
OpenSearch and queue tests marked as integration tests and run them against disposable containers.
Never commit `.env`, source blobs, traces containing content, model caches, or service keys.

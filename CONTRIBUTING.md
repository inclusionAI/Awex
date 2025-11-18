# How to contribute to Awex

## Finding good first issues

See [Good First Issues](https://github.com/inclusionAI/awex/contribute).

## How to create an issue

Create an issue with [this form](https://github.com/inclusionAI/awex/issues/new/choose).

## How to title your PR

Generally we follows the [Conventional Commits](https://www.conventionalcommits.org/) for pull request titles,
since we will squash and merge the PR and use the PR title as the first line of commit message.

For example, here are good PR titles:

- feat: support xxx feature
- fix: blablabla
- chore: remove useless yyy file
- docs: add api doc for xxx method

For more details, please check [pr-lint.yml](./.github/workflows/pr-lint.yml).

## Testing

For environmental requirements, please check [DEVELOPMENT.md](DEVELOPMENT.md).

### Python

Install test dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -v -s .
```

## Code Style

Run all checks: `bash ci/format.sh --all`.

### License headers

```bash
docker run --rm -v $(pwd):/github/workspace ghcr.io/korandoru/hawkeye-native:v3 format
```

### Python

Install formatting tools:

```bash
pip install ruff
```

Format Python code:

```bash
ruff format .
ruff check --fix .
```

### Markdown

```bash
npm install -g prettier
prettier --write "**/*.md"
```

## Development

For more information, please refer to [Development Guide](DEVELOPMENT.md).

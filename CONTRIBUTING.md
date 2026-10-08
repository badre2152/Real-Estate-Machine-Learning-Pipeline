# Contribution Guide

This repository contains a portfolio machine learning project. Changes are reviewed manually.

## Git workflow

The base branch is `master`. Create a working branch for changes and open a pull request before merging.

```bash
git clone https://github.com/badre2152/Real-Estate-Machine-Learning-Pipeline.git
cd Real-Estate-Machine-Learning-Pipeline
git checkout master
git pull origin master
git checkout -b feature/description
```

## Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your own database credentials and API key. Never commit real credentials, generated datasets or runtime logs.

## Code quality

Keep changes focused, preserve existing behavior and document any operational limitations. Use readable Python, descriptive variable names and useful logging that does not expose secrets.

The repository does not have automated CI or an automated test requirement. There is no need to add a test suite as part of routine documentation or maintenance updates. Runtime behavior must not be claimed as verified unless it was checked manually.

## Pull requests

Describe what changed, why it changed and any known limitations. Do not merge until changes have been reviewed and explicitly approved.

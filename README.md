# Project Setup

This project uses a Python virtual environment to manage dependencies.

---

## Getting Started

### 1. Create the virtual environment

Run this once when setting up the project for the first time:

```bash
python -m venv venv
```

### 2. Activate the virtual environment

Run this every time you return to work on the project:

```bash
source venv/bin/activate
```

> On Windows, use `venv\Scripts\activate` instead.

---

## Managing Dependencies

### Install a package

```bash
pip install requests
```

### Save installed packages to `requirements.txt`

```bash
pip freeze > requirements.txt
```

### Install all dependencies from `requirements.txt`

Useful when cloning the project or setting up a new environment:

```bash
pip install -r requirements.txt
```

### Install additional packages

```bash
pip install python-dotenv
```

> Don't forget to run `pip freeze > requirements.txt` again after installing new packages.

---

## Deactivating the Virtual Environment

When you're done working, deactivate the environment with:

```bash
deactivate
```
# rfml-dev

```bash
poetry install
poetry shell
jupyter notebook --ip 0.0.0.0
```



Poetry things to try:

```bash
poetry config virtualenvs.create true --local
```

Get the Poetry virtual env path, useful if you want to use the Poetry kernel in VS Code Notebook:

```bash
poetry env info --path
```

### Dependencies

You may need to install some CUDA libraries:

```bash
sudo apt install libcublas-12-0
```
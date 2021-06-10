# CNN with a Pretermined Fraction of Activations

## Requirements

Install the requirements:

```bash
pip install -r requirements.txt
```

Then add current directory to `$PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/containing/project/directory"
```

Following commands assume the name of the folder is `constant-fraction-activation`.

## Configuration Files

`config.toml` and `parameters.py` contain the hyperparameters and other configuration settings related to the project. Settings that change less frequently are stored in `config.toml`. Settings that change more frequently are stored in `parameters.py` and can be specified as command line arguments when any python file is run.

## License

Apache License 2.0

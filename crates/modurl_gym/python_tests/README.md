# Gymnasium parity fixtures

From the `modurl_gym` crate directory, regenerate one environment's fixtures
and run only its Rust parity tests:

```powershell
python python_tests/parity.py cartpole
```

Omit the environment to regenerate and test every registered environment:

```powershell
python python_tests/parity.py
```

Use `--rust-only` to test committed fixtures without Python dependencies, or
`--generate-only` to refresh fixtures without running Cargo. List supported
environment names with `python python_tests/parity.py --list`.

Fixture generation requires Gymnasium 1.2.1. LunarLander and BipedalWalker also
require box2d-py 2.3.5; install the matching extras with:

```powershell
python -m pip install "gymnasium[box2d]==1.2.1"
```

Ordinary Rust tests read the committed JSON fixtures and never invoke Python.

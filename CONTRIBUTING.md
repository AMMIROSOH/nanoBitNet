# Contributing

`nanoBitNet` optimizes for correctness, readability, and teaching value.

Good contributions:

- clarify BitNet math without adding framework complexity
- add tests for quantization behavior
- add small benchmarks with exact hardware details
- add educational notebooks or diagrams
- improve dataset loaders while keeping the default quick start simple

Please avoid:

- claiming production inference performance
- adding large dependencies for small conveniences
- hiding the quantization mechanics behind abstraction layers
- submitting benchmark numbers without device, dtype, and command details

Run before opening a PR:

```bash
pytest -q
python bench.py --steps 10
```

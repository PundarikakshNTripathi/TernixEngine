# Contribution Guidelines for TernixEngine

Thank you for your interest in contributing to TernixEngine! We welcome contributions from the community. By contributing, you agree that your contributions will be licensed under the Elastic License 2.0 (see `LICENSE` for details).

## Development Workflow
1. Fork the repository and create your branch from `develop`.
2. Ensure your code strictly adheres to C++20 standards.
3. Write GoogleTest unit tests for any new kernels or functions in the `tests/` directory.
4. Run the CI/CD pipeline locally via `run_all.bat` to ensure all tests and benchmarks pass.
5. Create a Pull Request targeting the `develop` branch.

## Coding Standards
- **No Emojis**: Keep documentation, commits, and comments professional and clinical.
- **Atomic Commits**: Follow Conventional Commits guidelines (e.g., `feat: ...`, `fix: ...`, `chore: ...`).
- **Dependencies**: TernixEngine is dependency-free by design. Do not introduce new external libraries for core inference logic.

## Reporting Bugs
When opening an issue for bug reports or performance degradation, please include:
- Your exact OS, CPU, and GPU models.
- The output of the failing command.
- Your `build/CMakeCache.txt` configuration file.

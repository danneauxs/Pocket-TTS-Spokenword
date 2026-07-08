# Development Workflow

## Testing
- Run tests: `pytest`
- Run specific test: `pytest test_file.py::test_function`
- Stop on first failure: `pytest -x`
- Verbose output: `pytest -v`

## Code Quality
- Format code: `black .`
- Lint code: `ruff check .`
- Type check: `mypy .` (if using type hints)

## Running the Application
- Main GUI: `python ASR.py`
- MFA Validator: `python mfa.py`
- Review Tool: `python review.py`

## Development Tips
- Always run tests before committing
- Use the ai_instant/ directory for quick questions
- Check code_patterns.md for common implementation patterns

# Contributing to Sign Language Translator

Thank you for your interest in contributing to the Sign Language Translator project! This document provides guidelines and information for contributors.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Sign_Language_Translator.git
cd Sign_Language_Translator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

## Project Structure

```
Sign_Language_Translator/
├── sign_language_translator/
│   └── sign_language_translator/     # Main package
│       ├── __init__.py              # Package exports
│       ├── datasets.py              # Dataset utilities
│       ├── models.py                # Model definitions
│       ├── transforms.py            # Data augmentation
│       ├── utils.py                 # Training utilities
│       └── scripts/                 # Command-line tools
│           ├── train.py
│           ├── infer.py
│           └── convert_sign_mnist.py
├── pyproject.toml                   # Package configuration
├── example_usage.py                 # Example code
└── README.md                        # Documentation
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep lines under 120 characters (see `pyproject.toml`)

Format code with Black (if installed):
```bash
black sign_language_translator/
```

## Making Changes

1. Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and test them:
```bash
python example_usage.py
```

3. Commit your changes with clear, descriptive messages:
```bash
git add .
git commit -m "Add feature: description of what you added"
```

4. Push to your fork and submit a pull request:
```bash
git push origin feature/your-feature-name
```

## Adding New Features

### Adding a New Backbone

To add a new model backbone (e.g., a new CNN architecture):

1. Update the `BackboneName` type in `models.py`
2. Add a new conditional block in `build_classifier()` function
3. Follow the existing pattern for other backbones
4. Update the README with the new option

### Adding New Data Augmentations

1. Add new transform functions in `transforms.py`
2. Integrate them into `build_train_transforms()` or `build_eval_transforms()`
3. Document the new augmentations

### Adding New Datasets

1. Create a new dataset class in `datasets.py` inheriting from `torch.utils.data.Dataset`
2. Add a configuration dataclass if needed
3. Add a builder function following the pattern of `build_imagefolder_dataset()`

## Testing

While this project doesn't currently have a comprehensive test suite, you should:

1. Run `example_usage.py` to verify basic functionality
2. Test all three CLI tools (`slt-train`, `slt-infer`, `slt-convert`) manually
3. Ensure your changes don't break existing functionality

## Documentation

- Update the README.md if you add user-facing features
- Add docstrings to new functions and classes
- Include examples in docstrings where helpful

## Questions?

Feel free to open an issue if you have questions or need help with your contribution!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

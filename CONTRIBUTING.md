# Contributing to Utrecht Housing Price Prediction

Thank you for considering contributing to this project! 🎉

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

Open an issue with:
- Clear description of the enhancement
- Why it would be useful
- Examples of how it would work

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Make your changes**
4. **Add tests if applicable**
5. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
6. **Push to the branch** (`git push origin feature/AmazingFeature`)
7. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/utrecht-housing-prediction.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/cla_project.ipynb
```

## Code Style

- Follow PEP 8 style guide
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

## Priority Areas

1. **Data Enhancement**
   - Add location coordinates
   - Collect additional property features
   - Integrate external datasets

2. **Model Improvement**
   - Implement cross-validation
   - Hyperparameter tuning
   - Feature engineering

3. **Documentation**
   - Add more visualizations
   - Improve code comments
   - Create tutorials

4. **Testing**
   - Unit tests for preprocessing
   - Model validation tests
   - Data quality checks

## Questions?

Feel free to open an issue for discussion or reach out directly!

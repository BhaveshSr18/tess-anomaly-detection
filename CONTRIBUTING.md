# Contributing to TESS Anomaly Detection Pipeline

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## 🎯 Code of Conduct

- Be respectful and inclusive
- Focus on scientific rigor and reproducibility
- Welcome diverse perspectives
- Provide constructive feedback

## 📋 How to Contribute

### 1. Report Bugs
Found a bug? Please open an issue with:
- **Title:** Clear description of the problem
- **Description:** What happened, what did you expect?
- **Reproduction:** Steps to reproduce the issue
- **Environment:** Python version, OS, GPU/CPU
- **Screenshots/Logs:** If applicable

### 2. Suggest Enhancements
Have a feature idea? Open an issue with:
- **Title:** Feature summary
- **Description:** Why is this needed? What problem does it solve?
- **Proposed Solution:** How would you implement it?
- **Alternatives:** Other approaches you considered

### 3. Submit Code Changes

#### Fork and Clone
```bash
git clone https://github.com/BhaveshSr18/tess-anomaly-detection.git
cd tess-anomaly-detection
git remote add upstream https://github.com/ORIGINAL_OWNER/tess-anomaly-detection.git
```

#### Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
git pull upstream main
```

#### Make Changes
- Write clean, documented code
- Follow PEP 8 style guidelines
- Add comments explaining complex logic
- Test your changes thoroughly

#### Code Style
```python
# Good: Clear and documented
def compute_anomaly_score(errors, periods, gaia_flag):
    """
    Compute multi-criterion astrophysical score.
    
    Parameters:
    -----------
    errors : np.ndarray
        Reconstruction errors (0-1)
    periods : np.ndarray
        Lomb-Scargle periods (days)
    gaia_flag : bool
        Gaia variability flag
        
    Returns:
    --------
    score : float
        Combined astrophysical score (0-1)
    """
    return (0.30 * errors + 
            0.20 * periods + 
            0.15 * gaia_flag)
```

#### Format and Lint
```bash
black notebooks/ src/ models/
flake8 src/ models/
```

#### Test Changes
```bash
# Create a test script or modify existing notebook
python -m pytest tests/
```

#### Commit Messages
```bash
# Good commit messages follow this format:
git commit -m "Add masked loss computation for padded light curves

- Implement MaskedMSELoss class
- Exclude padding regions from loss calculation
- Achieve 31x improvement on validation set
- Update documentation with new loss function"
```

#### Push and Create Pull Request
```bash
git push origin feature/your-feature-name
# Open PR on GitHub with:
# - Clear title
# - Description of changes
# - Link to relevant issues
# - Summary of testing performed
```

## 🔧 Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Install pre-commit hooks (optional)
pre-commit install
```

## 📝 Contribution Areas

### High Priority
- [ ] Extend to larger TESS datasets (>1000 sources)
- [ ] GPU optimization and benchmarking
- [ ] Additional loss functions (Huber, Quantile)
- [ ] Spectroscopic follow-up utilities

### Medium Priority
- [ ] More astronomical catalogs (ZTF, ASAS-SN)
- [ ] Interactive visualization dashboard
- [ ] Automated anomaly typing
- [ ] Statistical validation improvements

### Nice to Have
- [ ] GitHub Actions CI/CD pipeline
- [ ] Docker containerization
- [ ] Web interface for results exploration
- [ ] Comparative study with other anomaly methods

## 🧪 Testing Guidelines

### Unit Tests
```python
# tests/test_preprocess.py
import pytest
from src.preprocess import normalize_lightcurve

def test_normalize_lightcurve():
    """Test light curve normalization."""
    lc = np.array([1, 2, 3, 4, 5])
    normalized = normalize_lightcurve(lc)
    
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert len(normalized) == len(lc)
```

### Notebook Testing
- Run each notebook cell independently
- Verify output matches expected results
- Test with different random seeds
- Document any known failures

### Performance Testing
```bash
# Time the full pipeline
time jupyter nbconvert --to notebook --execute notebooks/04_train_autoencoder.ipynb

# Profile memory usage
python -m memory_profiler scripts/analyze.py
```

## 📊 Documentation Standards

### Code Documentation
```python
def compute_lombscargle_period(time, flux):
    """
    Compute Lomb-Scargle periodogram and find dominant period.
    
    This function computes the normalized Lomb-Scargle periodogram
    for a light curve and identifies the period with maximum power.
    
    Parameters
    ----------
    time : np.ndarray
        Time array (days), shape (N,)
    flux : np.ndarray
        Flux array (normalized), shape (N,)
        
    Returns
    -------
    period : float
        Dominant period (days)
    power : float
        Normalized power at dominant frequency (0-1)
        
    Notes
    -----
    Frequency oversampling factor set to 5 for fine resolution.
    Power threshold for detection: 0.1 (normalized Lomb-Scargle)
    
    References
    ----------
    Lomb, N. R. (1976). Least-squares frequency analysis of 
        unequally spaced data. Astrophysics and Space Science, 39, 447-462.
    Scargle, J. D. (1982). Studies in astronomical time series 
        analysis. Astrophysical Journal, 263, 835-853.
    """
    # Implementation...
```

### Markdown Documentation
- Use clear headers with proper hierarchy
- Include code examples
- Add badges for status/version
- Keep formatting consistent

## 🔄 Review Process

1. **Automated Checks**
   - Code style (flake8)
   - Tests pass
   - Documentation builds

2. **Manual Review**
   - Scientific correctness
   - Code quality and efficiency
   - Documentation clarity
   - Reproducibility

3. **Approval and Merge**
   - Approved by maintainers
   - All comments addressed
   - PR merged to main

## 🚀 Release Process

```bash
# Update version number
# Update CHANGELOG.md
# Create release branch
git checkout -b release/v1.1.0

# Tag release
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin release/v1.1.0
git push origin --tags
```

## 📚 Resources

- [Astropy Documentation](https://docs.astropy.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Lomb-Scargle Periodogram](https://en.wikipedia.org/wiki/Lomb%E2%80%93Scargle_periodogram)
- [TESS Mission](https://tess.mit.edu/)

## ❓ Questions?

- Open a GitHub Discussion
- Check existing issues for similar questions
- Review the documentation in `ANALYSIS_PIPELINE.md`
- Contact project maintainers

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under its MIT License.

---

**Thank you for contributing! 🙏**

Your contributions help advance open science and make this project better for everyone.

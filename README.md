# Praesepe Cluster Analysis: Membership, Tidal Structure, and Mass Segregation

## Overview

This project investigates the **Praesepe Cluster** (Messier 44) using data from **Gaia DR3** and advanced clustering techniques with **HDBSCAN**. The study focuses on three key scientific objectives:

1. **Membership Redetermination** - Accurately identify cluster members using astrometric and photometric data
2. **Tidal Structure** - Analyze the tidal boundaries and structural properties of the cluster
3. **Mass Segregation** - Examine how stellar mass correlates with spatial distribution within the cluster

## Project Description

The Praesepe Cluster is one of the nearest and best-studied open clusters, making it an ideal laboratory for understanding stellar dynamics and cluster evolution. This research leverages the unprecedented astrometric precision of Gaia DR3 combined with machine learning techniques to provide new insights into cluster membership and structure.

### Key Features

- **High-precision astrometry**: Utilizes Gaia DR3 parallax and proper motion measurements
- **Advanced clustering**: Implements HDBSCAN for robust cluster membership determination
- **Comprehensive analysis**: Examines spatial, kinematic, and physical properties
- **Statistical rigor**: Provides membership probabilities and uncertainty quantification

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- HDBSCAN
- Gaia data processing tools

## Installation

```bash
# Clone the repository
git clone https://github.com/chnoorlee/Astronomy.git
cd Astronomy

# Install required dependencies
pip install -r requirements.txt
```

## Data Sources

- **Gaia DR3**: Astrometric and photometric data from the European Space Agency's Gaia mission
- Position: RA ~08:40, Dec ~+20°
- Distance: ~182 pc

## Usage

[Add specific instructions for running analyses and generating results]

```python
# Example usage
import numpy as np
import hdbscan

# Load data and perform clustering analysis
# [Detailed code examples to be added]
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── [Gaia DR3 data files]
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   └── analysis.py
└── notebooks/
    └── [Jupyter notebooks for analysis and visualization]
```

## Results

[Summary of key findings regarding membership, tidal structure, and mass segregation]

## References

- Gaia Collaboration (2023) - Gaia DR3 documentation
- Kampakoglou et al. (2023) - Recent Praesepe studies
- McInnes et al. (2017) - HDBSCAN clustering algorithm

## Author

**Author**: [Your Name]  
**Email**: [Your Email]  
**Affiliation**: [Your Institution]

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Citation

If you use this analysis in your research, please cite:

```bibtex
@software{praesepe_analysis_2026,
  author = {[Author Name]},
  title = {Membership Redetermination, Tidal Structure, and Mass Segregation of the Praesepe Cluster Using Gaia DR3 and HDBSCAN},
  year = {2026},
  url = {https://github.com/chnoorlee/Astronomy}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or collaborations, please reach out through the GitHub repository or contact the author directly.

---

**Last Updated**: 2026-04-30

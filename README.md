## Privacy-Preserving Query Mechanisms for Spatiotemporal Data
### Project Overview
This project investigates and compares multiple privacy-preserving query mechanisms for spatiotemporal data. We analyze the effectiveness, efficiency, and trade-offs between utility and privacy for various techniques including K-anonymity, Laplace noise, temporal cloaking, and a novel combined approach.
### Motivation
As location-based services become increasingly prevalent, the need to protect user privacy while maintaining data utility has become critical. Spatiotemporal data is particularly sensitive as it can reveal personal information, habits, and behaviors when analyzed over time.
### Privacy Mechanisms Investigated
**K-anonymity**
Ensures that for any query result, the data of at least K different individuals are indistinguishable from each other, preventing the identification of specific individuals within the dataset.
**Laplace Noise**
Implements differential privacy by adding calibrated Laplace-distributed noise to query results, providing mathematical privacy guarantees while maintaining statistical accuracy.
**Temporal Cloaking**
Delays or aggregates data over time periods to obscure exact temporal information, reducing the precision of time-based inferences.
**Combined Approach (Laplace Noise + K-anonymity)**
A novel hybrid solution that leverages the strengths of both K-anonymity and Laplace noise to provide strong privacy guarantees while minimizing utility loss.
### Implementation

- All mechanisms implemented in Python
- Visualization created using Matplotlib
- Synthetic dataset generation for comprehensive testing
- Performance metrics tracked across different privacy parameters

### Methodology

- Generated synthetic spatiotemporal datasets with controlled properties
- Implemented each privacy mechanism in Python
- Evaluated mechanisms across multiple metrics:

  - Privacy protection (re-identification risk)
  - Query accuracy
  - Computational efficiency
  - Scalability with dataset size


Performed comparative analysis to identify optimal approaches for different use cases

### Results
Our analysis provides insights into which privacy mechanisms are most suitable for different types of spatiotemporal queries, data distributions, and privacy requirements. The combined approach shows promising results in balancing privacy and utility across most test scenarios.
### Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

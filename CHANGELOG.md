
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [this sample changelog](https://gist.github.com/juampynr/4c18214a8eb554084e21d6e288a18a2c).
 
## [0.2.0] - 2023-02-dd
 
 - Changelog
### Added
- VAE, GCN, SAGE and GAT models based on tensorflow (VAEG code)
- SCG-based loss to train VAE and GNNs
- Output assembly stats while starting
- Eliminate VAMB and DGL dependencies
- PyPI installation
 
### Changed
- Code structure changed to load data outside of DGL and use DGL only for the GraphSAGE-LSTM model
- Log dataloading steps
- Write cache to numpy files
 
### Fixed
- Feature files are written to specific directies (fixes #17)
 
## [0.1.3] - 2022-02-25

BioarXiv version
  
`pip install . --upgrade`
 
### Added
- Dockerfile and docker image link
- Set seed option
- Eval interval option
 
### Changed
  
- Change default file name

 
### Fixed
 
- Assembly dir option is no longer mandatory, so files can be in different directories
- Logging also includes errors
- DGL should no longer write a file to ~/
 

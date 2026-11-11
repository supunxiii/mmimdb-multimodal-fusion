## Overview

MM-IMDb Multimodal Fusion is a multimodal machine learning project for multi-label movie genre classification using the MM-IMDb dataset. The project investigates how visual information from movie posters and textual information from plot summaries can be combined to improve genre prediction.

The implementation compares unimodal baselines with multiple multimodal fusion strategies, including Early Fusion, Late Fusion, and CLIP-based fusion. The final system evaluates 23 movie genres using multiple performance metrics, threshold optimisation, per-genre analysis, generalisation assessment, and error analysis.

The project also includes initial and subsequent dataset-analysis notebooks, data-quality assessment, feature preprocessing, model evaluation, ethics and bias assessment, and reproducibility documentation.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-412991?style=flat-square)
![h5py](https://img.shields.io/badge/h5py-HDF5-0F6B8A?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-MM--IMDb-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-Report-008080?style=flat-square&logo=latex&logoColor=white)

## Features

MM-IMDb Multimodal Fusion provides the following features:

1. **Initial Dataset Investigation**

   - Analysed the MM-IMDb dataset before model development to assess its suitability for multimodal classification
   - Examined dataset size, available modalities, genre labels, data structure, and class distribution
   - Verified that movie posters and plot summaries could be aligned for multimodal learning

2. **Detailed Data Quality Assessment**

   - Performed a subsequent implementation-stage analysis of the supplied MM-IMDb binary files
   - Verified tokenised plot summaries by decoding sequences using the supplied vocabulary
   - Reconstructed and visually inspected movie posters
   - Checked IMDb identifier uniqueness and target-label integrity
   - Analysed genre frequency and multi-label class imbalance

3. **Train, Validation, and Test Partitioning**

   - Used multi-label stratification to construct training, validation, and test partitions
   - Created a 70/15/15 split containing 18,148 training, 3,908 validation, and 3,903 test observations
   - Verified that there was no overlap between partitions
   - Confirmed that genre prevalence remained highly consistent across all partitions

4. **Feature Preprocessing**

   - Processed the supplied 4,096-dimensional VGG visual features
   - Removed 13 zero-variance visual dimensions, leaving 4,083 usable features
   - Used the supplied 300-dimensional textual representation
   - Standardised visual and textual features independently using training-set statistics only
   - Checked features for missing and infinite values

5. **Unimodal Baseline Models**

   - Developed separate Visual Only and Text Only genre-classification models
   - Formulated the problem as 23 one-vs-rest binary classification tasks
   - Used stochastic gradient descent with logistic loss
   - Applied balanced class weighting to reduce the effect of genre imbalance

6. **Early Fusion**

   - Concatenated the cleaned visual and textual representations
   - Created a 4,383-dimensional multimodal feature representation
   - Trained a shared classifier using information from both modalities simultaneously

7. **Late Fusion**

   - Retained independent visual and textual classifiers
   - Combined their predicted probabilities using weighted decision-level fusion
   - Optimised modality weights using the validation partition
   - Selected final weights of 0.40 for visual predictions and 0.60 for textual predictions

8. **CLIP-Based Multimodal Fusion**

   - Used the pretrained `openai/clip-vit-base-patch32` model
   - Extracted 512-dimensional poster embeddings
   - Extracted 512-dimensional plot-summary embeddings
   - Applied L2 normalisation to CLIP representations
   - Concatenated the image and text embeddings into a 1,024-dimensional multimodal representation
   - Cached extracted CLIP features to avoid repeated feature extraction

9. **Validation and Threshold Optimisation**

   - Used the validation partition for model selection and optimisation
   - Optimised classification thresholds using Macro F1-score
   - Refined the threshold search when the Early Fusion optimum occurred near the initial search boundary
   - Kept the test partition untouched until final model evaluation

10. **Comprehensive Model Evaluation**

   - Evaluated models using Micro F1, Macro F1, Weighted F1, Micro Precision, Micro Recall, and Hamming Loss
   - Compared Visual Only, Text Only, Early Fusion, Late Fusion, and CLIP Fusion approaches
   - Performed per-genre precision, recall, F1-score, and support analysis
   - Investigated strongest and weakest genre predictions
   - Analysed validation-to-test generalisation to assess overfitting

11. **Error and Generalisation Analysis**

   - Examined class-specific errors across all 23 genres
   - Investigated the relationship between class support and classification performance
   - Compared validation and test Macro F1-scores
   - Identified remaining weaknesses for difficult or minority genres

12. **Ethics and Bias Assessment**

   - Considered genre imbalance, representation bias, and potential multimodal bias amplification
   - Discussed possible biases inherited from pretrained CLIP representations
   - Considered privacy, transparency, accountability, and deployment limitations
   - Evaluated fairness primarily through class-specific performance because demographic attributes were unavailable

13. **Final Model Selection**

   - Selected CLIP Fusion as the final approach
   - Achieved a test Micro F1-score of 0.5719
   - Achieved a test Macro F1-score of 0.5676
   - Achieved a test Weighted F1-score of 0.5733
   - Achieved the lowest Hamming Loss of 0.0848
   - Demonstrated stable validation-to-test generalisation

## Technologies Used

- **Python**: Core programming language
- **Google Colab**: Cloud-based notebook and model-development environment
- **Jupyter Notebook**: Exploratory analysis and implementation workflow
- **NumPy**: Numerical processing and feature manipulation
- **Pandas**: Data organisation, analysis, and result tables
- **Matplotlib**: Data visualisation and model-performance plots
- **h5py**: Reading and processing the MM-IMDb HDF5 dataset
- **scikit-learn**: Classification, preprocessing, evaluation metrics, and model utilities
- **iterative-stratification**: Multi-label stratified train, validation, and test splitting
- **PyTorch**: Deep-learning backend for CLIP feature extraction
- **Hugging Face Transformers**: Pretrained CLIP model and processor
- **CLIP ViT-B/32**: Pretrained image-text representation model
- **Google Drive**: Dataset, feature-cache, and experiment storage
- **Kaggle**: Source of the MM-IMDb dataset distribution
- **Git**: Version control
- **GitHub**: Source-code hosting and reproducibility documentation
- **LaTeX**: Final academic report preparation

## Project Specifications

- **Project Type**: Multimodal multi-label classification
- **Dataset**: MM-IMDb
- **Total Movies**: 25,959
- **Number of Genres**: 23
- **Input Modalities**: Movie posters and plot summaries
- **Training Samples**: 18,148
- **Validation Samples**: 3,908
- **Test Samples**: 3,903
- **Visual Feature Dimension**: 4,083 after preprocessing
- **Text Feature Dimension**: 300
- **Early Fusion Dimension**: 4,383
- **CLIP Image Embedding Dimension**: 512
- **CLIP Text Embedding Dimension**: 512
- **CLIP Fusion Dimension**: 1,024
- **Classification Strategy**: One-vs-Rest multi-label classification
- **Fusion Approaches**: Early Fusion, Late Fusion, and CLIP Fusion
- **Final Selected Model**: CLIP Fusion
- **Primary Selection Metric**: Macro F1-score

## Getting Started

To reproduce the MM-IMDb Multimodal Fusion project, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/supunxiii/mmimdb-multimodal-fusion.git
   ```

2. Navigate to the project directory:

   ```shell
   cd mmimdb-multimodal-fusion
   ```

3. Install the required Python dependencies:

   ```shell
   pip install numpy pandas matplotlib h5py scikit-learn iterative-stratification torch transformers
   ```

4. Download the MM-IMDb dataset from Kaggle:

   ```text
   https://www.kaggle.com/datasets/johnarevalo/mmimdb
   ```

5. The implementation expects the following MM-IMDb dataset files:

   ```text
   metadata.npy
   multimodal_imdb.hdf5
   ```

6. When using Google Colab, place the extracted dataset in the configured project directory on Google Drive. The implementation used:

   ```text
   /content/drive/MyDrive/MMIMDb_Project/mmimdb_extracted/
   ```

7. Open the analysis notebooks to review the initial and subsequent dataset investigations before running the final implementation.

8. Open the main implementation notebook:

   ```text
   MMIMDb_multimodal_fusion.ipynb
   ```

9. Run the notebook cells sequentially. The workflow performs:

   ```text
   Dataset loading
        ↓
   Data-quality assessment
        ↓
   Train / validation / test partitioning
        ↓
   Feature preprocessing
        ↓
   Unimodal baseline training
        ↓
   Early and Late Fusion
        ↓
   CLIP feature extraction
        ↓
   CLIP Fusion training
        ↓
   Validation threshold optimisation
        ↓
   Final test evaluation
        ↓
   Per-genre and error analysis
   ```

10. CLIP embeddings are cached after extraction so that subsequent runs can reuse the generated feature files instead of processing all movie posters and plot summaries again.

## Project Structure

```text
mmimdb-multimodal-fusion/
├── analysis/                       # Initial and subsequent dataset-analysis notebooks
├── MMIMDb_multimodal_fusion.ipynb # Main multimodal modelling notebook
├── figures/                        # Generated evaluation figures
├── results/                        # Model evaluation outputs
├── README.md                       # Project documentation
└── LICENSE                         # Licence
```

Large MM-IMDb dataset files and generated CLIP feature caches are not intended to be committed directly to the repository. They should be downloaded or generated locally according to the instructions above.

## Developer

This project was developed by:

- **Supun Wijesooriya** - Developer

## Contributing

Contributions are welcome. If you would like to contribute:

1. Fork the repository.

2. Create a new branch:

   ```shell
   git checkout -b feature/your-feature-name
   ```

3. Commit your changes:

   ```shell
   git commit -m "Add your commit message"
   ```

4. Push your branch:

   ```shell
   git push origin feature/your-feature-name
   ```

5. Open a pull request with a clear description of your changes.

## License

This project is open-source and is licensed under the [MIT License](LICENSE).

The MM-IMDb dataset and externally sourced pretrained models remain subject to their respective licences and usage conditions.

## Contact

For any enquiries or feedback, please contact the developer:

- **Supun Wijesooriya**: [GitHub Profile](https://github.com/supunxiii)
- **Project Repository**: [mmimdb-multimodal-fusion](https://github.com/supunxiii/mmimdb-multimodal-fusion)

---

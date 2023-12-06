<div id="top"></div>

<div align="center">
  <h1>
    SynthCheck: a dashboard to evaluate synthetic data quality
  </h1>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#application-structure">Application Structure</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project

Machine Learning and Artificial Intelligence are increasingly being exploited to solve health-related problems, such as prognosis prediction from Electronic Health Records or detecting patterns in multi-omics data. Data plays a significant role in the development of such systems, but concerns have been raised when dealing with patient's data, with regulators underlying the need to protect patients' privacy. To this end, in recent years, there has been a growing proposal to replace original data (derived from real patients) with the use of synthetic data that mimic the main statistical characteristics of their real counterparts.
Regardless of the methods employed to generate them, it is essential to assess the quality of the synthetic data. To address this constraint, we've created a Dash application that users can install and utilize on their computers. This application allows users to upload both original and synthetic data, generating various metrics to assess resemblance, utility, and privacy. Furthermore, users can download a report containing the obtained results.

<p align="right"><a href="#top">↰ Back To Top</a></p>

## Installation

This repository provides a Conda environment configuration file (`synthcheck_env.yml`) to streamline the setup process. Follow these steps to create the environment:

> [!IMPORTANT]
> Make sure you have Conda installed. If not, [install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) before proceeding.

### Steps to Create the Environment

1. **Create the Conda Environment**

    Run the following command to create the environment using the provided `.yml` file:

    ```bash
    conda env create -f synthcheck_env.yml
    ```

    This command will set up a Conda environment named according to specifications in the `synthcheck_env.yml` file.

2. **Activate the Environment**

    Once the environment is created, activate it using:

    ```bash
    conda activate synthcheck_env
    ```

### Running the Code

Once the virtual environment is activated, you can run the code using the following steps:

```bash
python SynthCheck_app.py
```

### Additional Notes

- To deactivate the environment, simply use:

    ```bash
    conda deactivate
    ```

- You can now work within this Conda environment to run the application.

<p align="right"><a href="#top">↰ Back To Top</a></p>

## Application Structure

The application is organized into two main sections:

### Data Upload for Quality Evaluation

The data upload process for quality evaluation is divided into several components:

#### 1. Uploading Original and Synthetic Datasets
Users are prompted to upload two CSV files:
- **Original Dataset**: it contains the dataset used when generating the synthetic data ([example original dataset](example%20datasets/original_dataset.csv)).
- **Synthetic Dataset**: it comprises the synthetic data for quality evaluation purposes ([example synthetic dataset](example%20datasets/synthetic_dataset.csv)).

> [!TIP]
> Ensure that categorical feature categories are encoded with numerical values (e.g., 'benign' = 0 and 'malign' = 1).

#### 2. Feature Type Descriptor File
In addition to the datasets, users are required to upload a descriptor file in CSV format ([example feature type file](example%20datasets/features_types.csv)). This file is structured with two columns:

##### Example:

| Feature         | Type       |
|-----------------|------------|
| Age             | numerical  |
| Gender          | categorical|
| Income          | numerical  |
| Education       | categorical|

> [!WARNING]
> The accepted values in the 'Type' column are exclusively 'numerical' and 'categorical'. Additionally, the file must include column headers.

### Quality Assessment of Synthetic Data

The second section empowers users to perform a comprehensive quality assessment of the uploaded synthetic data. This section comprises three subsections, each dedicated to implementing distinct quality analyses.

#### Resemblance Section

This section provides access to three subsections:

1. **URA Analysis**: it conducts various statistical tests and distance metric comparisons for both numerical and categorical features.

2. **MRA Analysis**: it omputes metrics related to Multiple Resemblance Analysis such as correlation matrices, outliers analysis, variance explained analysis and UMAP method implementations.

3. **DLA Analysis**: it presents, for each classifier used in the Data Labeling Analysis, the values of performance metrics.

#### Utility Section

This section implements TRTR (Train on Real, Test on Real) and TSTR (Train on Synthetic, Test on Real) approaches for a selected target class and machine learning model.

#### Privacy Section

This section consists of three subsections dedicated to privacy evaluation:

1. **SEA Analysis**: it computes metrics like cosine similarity, Euclidean distance and Hausdorff distance, displaying corresponding density plots or values.

2. **MIA Simulation**: it simulates Membership Inference Attacks with adjustable attacker parameters and showcases attacker performance.

3. **AIA Simulation**: it allows simulation of Attribute Inference Attacks where the user sets the attacker's access to features, displaying recostruction performance metrics.

Each section provides options to download reports containing the displayed graphs and tables.

<p align="right"><a href="#top">↰ Back To Top</a></p>

## License

Distributed under MIT License. See `LICENSE` for more information.

<p align="right"><a href="#top">↰ Back To Top</a></p>

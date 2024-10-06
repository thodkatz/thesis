<a name="readme-top"></a>

<h3 align="center">UnitedNet</h3>

  <p align="center">
    Explainable multi-task learning for multi-modality biological data analysis
    <br />
    <a href="https://www.nature.com/articles/s41467-023-37477-x"><strong>Explore the manuscript
</strong></a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#Compatibility with Google Colab">Compatibility with Google Colab</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Current biotechnologies can simultaneously measure multiple high-dimensional modalities (e.g., RNA, DNA accessibility, and proteins) from the same cells. A combination of different analytical tasks (e.g., multi-modal integration and cross-modal analysis) is required to comprehensively understand such data, inferring how gene regulation drives biological diversity and functions. However, current analytical methods are designed to perform a single task, only providing a partial picture of the multi-modal data. Here, we present UnitedNet, an interpretable multi-task deep neural network capable of integrating different tasks to analyze single-cell multi-modality data. Applied to various multi-modality datasets (e.g., Patch-seq, multiome ATAC+gene expression, and spatial transcriptomics), UnitedNet demonstrates similar or better accuracy in multi-modal integration and cross-modal prediction compared with state-of-the-art methods. Moreover, by dissecting the trained UnitedNet with the explainable machine learning algorithm, we can directly quantify the relationship between gene expression and other modalities with cell-type specificity. UnitedNet is a comprehensive end-to-end framework that will be broadly applicable to single-cell multi-modality biology, potentiating the discovery of cell-type-specific regulation kinetics across transcriptomics and other modalities.

![Alt text](./data/UnitedNet.jpg?raw=true "UnitedNet")

### Built With
* python 3.7
* pytorch 1.11
* jupyter notebook
* PyCharm


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

* we recommend to use GPU for faster training. However, when a GPU is not available,
please specify in the code with
  ```sh
  device = "cpu"
  ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
As for demonstration, we have provided four detailed example jupyter notebooks to reproduce the results in the manuscript.
* Please find all the notebooks to analyze the data in ./notebooks

Specifically, UnitedNet take the [AnnData](https://anndata.readthedocs.io/en/latest/) as input. It only takes few line of codes to run
the UnitedNet.
  ```sh 
  model = UnitedNet(save_path, device=device, technique=dlpfc_config)
  model.train(adatas_train)
  # for annotation transfer
  model.transfer(adatas_train, adatas_test)
  # for classification/clustering
  model.predict_label(adatas_train)
  # for multi-modal fusion
  adata_fused = model.infer(adatas_train)
  # for cross-modal prediction
  adatas_prd = model.predict(adatas_test)
  ```

Additionally, as a trained UnitedNet combines information for both multimodal group identification and cross-modal prediction, dissecting it using standard explainable machine learning methods can reveal the cell-type-specific, cross-modal feature-to-feature relevance, which can help to identify new biological insights from multimodal biological data. To do this, we apply the SHapley Additive exPlanations algorithm ([SHAP](https://github.com/slundberg/shap)), commonly used to interpret deep learning models, to dissect the trained UnitedNet. During the explainable learning, we can identify features that show higher relevance to specific groups and then quantify the cross-modal feature-to-feature relevance within these groups. It should be noted that, owing to the inherent randomness involved in both model training and the implementation of SHAP, the results identified by SHAP may exhibit some degree of variability. However, it is expected that the majority of the selected outcomes will remain consistent.
![Alt text](./data/explainablelearning.jpg?raw=true "UnitedNet")
  ```sh 
  # Dissecting the group identification module can enable a group-to-feature relevance analyses
  from unitednet.modules import submodel_clus
  sub = submodel_clus(model.model).to(model.device)
  # select a set of background examples to take an expectation over
  background = cluster_prototype_features
  e = shap.DeepExplainer(sub, background)
  # choose what to explain
  shap_values = e.shap_values(test_type,check_additivity=True)
  ```
  ```sh 
  # Further dissecting the cross-modal prediction module can enable a group-specific cross-modal feature-to-feature relevance analyses
  from unitednet.modules import submodel_trans
  sub_0_1 = submodel_trans(model.model,[0,1]).to(model.device)
  e_model_0_1 = shap.DeepExplainer(sub_0_1, background_0_1)
  shap_values_0_1 = e_model_0_1.shap_values(test_type_0_1,check_additivity=True)
  ```
For the cross-modal relevance analysis, please find more details in _./notebooks/PatchSeq-Relevance_Analysis.ipynb_ and
_./notebooks/ATACSeq-Relevance_Analysis.ipynb_


_For more implementation details, please refer to the manuscript_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Compatibility with Google Colab -->
## Compatibility with Google Colab
For better running of the code, we also recommend you to use [Google Colab](https://colab.research.google.com/) to explore the UnitedNet.
Google Colab support cloud computing with free GPU, which can significantly increase the training efficiency.
We have made our code compatible with Google Colab here. We note that because of the inherent randomness in deep neural network training,
the results could be slightly different as shown in the paper but should be mostly similar.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GPL-3.0 license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Xin Tang - xintang@g.harvard.edu

Jiawei Zhang - zhan4362@umn.edu

Yichun He - yichunhe@g.harvard.edu

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [datasets] Data was all from publicly available datasets and previous studies. The Dygen simulation data can be reproduced by the simulator in https://github.com/dynverse/dyngen. The MUSE simulation data can be reproduced by the simulator in https://github.com/AltschulerWu-Lab/MUSE. The original modality of MNIST data was downloaded from http://yann.lecun.com/exdb/mnist. The Patch-seq GABAergic neuron dataset was downloaded from https://github.com/AllenInstitute/coupledAE-patchseq and https://portal.brain-map.org. The ATAC-seq BMMC dataset was downloaded from https://openproblems.bio/neurips. The DBiT-dataset was downloaded from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137986. The DLPFC dataset was downloaded from https://doi.org/10.18112/openneuro.ds002076.v1.0.1. 
* We used [SHAP](https://github.com/slundberg/shap) for interpretable machine learning

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CITATIONS -->
## Citations
Please cite us if you find the code or manuscript is useful to you:

> Tang, X. et al. Explainable multi-task learning for multi-modality biological data analysis. Nature Communications 14, 2546 (2023). https://doi.org/10.1038/s41467-023-37477-x

<p align="right">(<a href="#readme-top">back to top</a>)</p>


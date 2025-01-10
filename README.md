Wouldn't it be nice, if you could...

# Predict weather from climate

This code accompanies the paper **"Spatiotemporally Coherent Probabilistic Generation of Weather from Climate"**, which can be found [here (arXiv)](https://arxiv.org/abs/2412.15361) ➡️ `https://arxiv.org/abs/2412.15361`

<p align="center">
  <img src="./assets/vas_storm.png" width="75%" />
</p>

This repository contains the code to reproduce the results of the paper.
Please find information on...
- ...data processing
- ...loading trained models
- ...loading results used in the paper
- ...training the model yourself
- ...etc.

...below!

## Have a look yourself: Climate-Informed Weather Dynamics 🌦️

The downscaled climate simulations for the year 2014, which are evaluated in the paper, are accessible [here (huggingface)](https://huggingface.co/datasets/schmidtjonathan/C2W_downscaled-2014/tree/main) ➡️ `https://huggingface.co/datasets/schmidtjonathan/C2W_downscaled-2014`

Please feel encouraged to load, analyze and use the climate-informed weather simulations!
- Are they useful for you and your research? Amazing! Please let me know and please cite our paper when you use our results 🚀
- Are there problems; either with loading the data or with using it? Do you have feedback? Please do not hesitate to let me know, e.g., by opening an issue in this repository! Thank you!
I am very interested in hearing your opinion!



## Convince yourself: load the trained model 💪

To reproduce all results from the experiments, please download the pickled model (and/or the training state) from [here (huggingface)](https://huggingface.co/schmidtjonathan/C2W_model/tree/main) ➡️ `https://huggingface.co/schmidtjonathan/C2W_model/tree/main`

Please feel encouraged to load the model and to use it for your own experiments! And do not hesitate to share your opinion or questions via e-mail or by opening an issue in this repository! Thank you!

---

### Acknowledgements

The model and code extends [Score-Based Data Assimilation](https://github.com/francois-rozet/sda) by François Rozet et al.
Some training utility is adopted from the [EDM2 Repository](https://github.com/NVlabs/edm2/tree/main) by Tero Karras et al.
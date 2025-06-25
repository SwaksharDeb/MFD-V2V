# Unsupervised Cardiac Video Translation Via Motion Feature Guided Diffusion Model
## Abstract

This paper presents a novel motion feature guided diffusion model for unpaired video-to-video translation (MFD-V2V), designed to synthesize dynamic, high-contrast cine cardiac magnetic resonance (CMR) from lower-contrast, artifact-prone displacement encoding with stimulated echoes (DENSE) CMR sequences. To achieve this, we first introduce a Latent Temporal Multi-Attention (LTMA) registration network that effectively learns more accurate and consistent cardiac motions from cine CMR image videos. A multi-level motion feature guided diffusion model, equipped with a specialized Spatio-Temporal Motion Encoder (STME) to extract fine-grained motion conditioning, is then developed to improve synthesis quality and fidelity. We evaluate our method, MFD-V2V, on a comprehensive cardiac dataset, demonstrating superior performance over the state-of-the-art in both quantitative metrics and qualitative assessments. Furthermore, we show the benefits of our synthesized cine CMRs improving downstream clinical and analytical tasks, underscoring the broader impact of our approach

## Our Approach: MFD-V2V

![MFD-V2D Architecture](figures/overall_fig.png)

## Demonstration of MFD-V2V
<p align="center">
  <img src="MFD-V2V_demos/A01_P101_evaluation.gif" width="250" />
  <img src="MFD-V2V_demos/A01_P104_evaluation.gif" width="250" />
  <img src="MFD-V2V_demos/A01_P12_evaluation.gif" width="250" />
</p>
<p align="center">
  <img src="MFD-V2V_demos/A01_P17_evaluation.gif" width="250" />
  <img src="MFD-V2V_demos/A01_P19_evaluation.gif" width="250" />
  <img src="MFD-V2V_demos/A01_P22_evaluation.gif" width="250" />
</p>
<p align="center">
  <img src="MFD-V2V_demos/A01_P25_evaluation.gif" width="250" />
  <img src="MFD-V2V_demos/A01_P34_evaluation.gif" width="250" />
  <img src="MFD-V2V_demos/A01_P35_evaluation.gif" width="250" />
</p>
<p align="center">
  <img src="MFD-V2V_demos/A01_P42_evaluation.gif" width="250" />
  <img src="MFD-V2V_demos/A01_P43_evaluation.gif" width="250" />
  <img src="MFD-V2V_demos/A01_P46_evaluation.gif" width="250" />
</p>

In the above demos, for each coloum, **(left)** is the DENSE CMR sequences and **(right)** is the paired Cine CMR sequences synthesized by our method MFD-V2V. 

## How to Train MFD-V2V
### Train the Registration Network
If you want to train the registration network, then go to the "Train LTMA registration Network" and run the Main.py file. The code will automatically save the last checkpoint for you, so you don't need to do anything. After training the registration network put the checkpoint in the models folder. We already provide the pretrain weights of the registration network inside the models folder.

Run this below command if you want to train our registration network

`python Train LTMA registration Network/Main.py`

### Train the Diffusion Model
If you want to train the diffusion model Run the below command

`python train_video_flow_diffusion.py file` 

We already provide the pretrain weight of our registration network (LTMA) as a .pth file inside the models folder.

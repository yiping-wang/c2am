# Causal Class Activation Maps (C^2AM) for Weakly-Supervised Semantic Segmentation

- We propose a Structural Causal Model for causal intervention to analyze the causalities among the *Style* and *Content* of images, image-level labels, and weak localization cues, by *front-door adjustment*. 
- Our algorithm, named Causal CAM (C^2AM), can mitigate the confounding bias in image-level classification, <u>without</u> any additional parameters or manipulation of the images, to produce high-quality weak localization cues. 
- We evaluated C^2AM PASCAL VOC 2012 and achieved mIOU 68.25% of the pseudo mask generation on the training set, and mIOU xx.xx% and xx.xx% on validation and test set when training DeepLabV2 on the seed masks.

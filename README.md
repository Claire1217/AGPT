# Anatomical Grounding Pre-training for Medical Phrase Grounding
## Abstract
Medical Phrase Grounding (MPG) maps radiological findings to specific regions in medical images, aiding in the understanding of radiology reports. Existing generaldomain models struggle in zero-shot and fine-tuning settings with MPG due to domain-specific challenges and limited annotated data. We propose anatomical grounding as an in-domain pre-training task that aligns anatomical terms with corresponding regions in medical images, leveraging large-scale datasets such as Chest ImaGenome. Our emperical evaluation on MS-CXR demonstrates that anatomical
grounding pre-training significantly improves performance in both a zero-shot and fine-tuning setting, outperforming selfsupervised models like GLoRIA and BioViL. Our fine-tuned
model achieved state-of-the-art performance on MS-CXR with an mIoU of 61.2, demonstrating the effectiveness of anatomical grounding pre-training for MPG.

## Description
This repository provides the code necessary to load the MDETR and TransVG models, which are pre-trained on the ChestImagenome anatomy grounding dataset and subsequently fine-tuned on the MS-CXR medical phrase grounding dataset. The demo folder includes four example Chest X-ray images from the MS-CXR dataset. The demo.py script supplies descriptive phrases along with their corresponding bounding box annotations for each image. Additionally, the repository offers visualizations that display the model's predictions, highlighting the annotated bounding boxes on the sample Chest X-ray images.

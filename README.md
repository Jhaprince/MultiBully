# A Multitask Framework for Sentiment, Emotion and Sarcasm aware Cyberbullying Detection in Multi-modal Code-Mixed Memes (SIGIR 2022)

This is the official repository accompanying the SIGIR 2022 full paper [A Multitask Framework for Sentiment, Emotion and Sarcasm aware Cyberbullying Detection in Multi-modal Code-Mixed Memes ](https://www.cse.iitb.ac.in/~pb/papers/sigir22-sa-multitask.pdf). This repository contains codebase, dataset and the annotation guidelines.

# Authors
Krishanu Maity*, Prince Jha*, Sriparna Saha, Pushpak Bhattacharyya

Note : * denotes equal contribution

# Dataset
1. [Text Annotated with Bully, Sentiment, Emotion, Sarcasm, Target, and Harfulness Score](https://docs.google.com/spreadsheets/d/11JSgF-ZoHOQXiT8aj4RnFNz97UiAM5Ql_0MEq_RNjik/edit?usp=sharing)
2. [Meme Image](https://drive.google.com/drive/folders/1_01joFDElDHGc47iU4QShoG1EDhvf6zM?usp=sharing)

# Model Training
1. Train the model using auxiliary tasks (Emotion, Sentiment, Sarcasm) and main task together with centralnet and simultaneosuly optimize loss function for all tasks
2. Although by training model on all tasks simultaneously makes the model overfit on training data because of complexity.
3. To overcome that we first train the model individually on all tasks using basic architecture given :
    (i) Bully
    (ii) Sentiment 
    (iii) Emotion
    (iv) Sarcasm
 4. Then you can load the pretrained weight from bully, sentiment, emotion and sarcasm and freeze the network parameters for these layers. and only learn the weights for Central Network to predict the output.
 5. [here](https://github.com/Jhaprince/MultiBully/blob/9a35c4cc7bfdcbea00a9661fbd77bff69fdc1e12/train.py#L258) you can decide which model should be used for training
 6. Here([Emotion](https://github.com/Jhaprince/MultiBully/blob/9a35c4cc7bfdcbea00a9661fbd77bff69fdc1e12/centralnet.py#L240),[Sarcasm](https://github.com/Jhaprince/MultiBully/blob/9a35c4cc7bfdcbea00a9661fbd77bff69fdc1e12/centralnet.py#L247),[Sentiment](https://github.com/Jhaprince/MultiBully/blob/9a35c4cc7bfdcbea00a9661fbd77bff69fdc1e12/centralnet.py#L253),[Bully](https://github.com/Jhaprince/MultiBully/blob/9a35c4cc7bfdcbea00a9661fbd77bff69fdc1e12/centralnet.py#L258)) you can freeze the network parameters or you can comment if you want to finetune it again for

# Annotation Guidelines
To be done

# References
I would encourage to read following papers within the given sequence before moving to the code part
1. [CLIP](https://arxiv.org/pdf/2103.00020.pdf)
2. [The Verbal and Non Verbal Signals of Depression--Combining Acoustics, Text and Visuals for Estimating Depression Level](https://arxiv.org/pdf/1904.07656.pdf)
3. [CentralNet: a Multilayer Approach for Multimodal Fusion](https://arxiv.org/pdf/1808.07275.pdf)
4. [MOMENTA](https://aclanthology.org/2021.findings-emnlp.379.pdf)

# Code 
We have borrowed our code from following sources, you can also go visit these
1. [CLIP Feature Extractor](https://github.com/openai/CLIP)
2. [centralnet code](https://github.com/jperezrua/mfas)
3. [MOMENTA](https://github.com/lcs2-iiitd/momenta)


# Citation
If you find this repository to be helpful please cite us

```
@article{maity2022multitask,
  title={A Multitask Framework for Sentiment, Emotion and Sarcasm aware Cyberbullying Detection from Multi-modal Code-Mixed Memes},
  author={Maity, Krishanu and Jha, Prince and Saha, Sriparna and Bhattacharyya, Pushpak},
  year={2022}
}
```


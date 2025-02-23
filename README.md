# README

# BERT-based Text Classification / 基于 BERT 的文本分类

## 项目概述 / Project Overview

- 本项目并非作者原创 / This project is not original by the author.
  Forked from [Git repository](https://github.com/wxxxuser/aliyuntianchi-ncaa2024), Author: wxxuer.

- 旨在学习与课程训练，在原仓库的基础上修改与合并，使得项目更条理清晰 / This project aims to learn and train for the course, with modifications and merges on the original repository to make the project more organized.

- Modified by: bebopoy, Email: xiangtongnie@gmail.com. 联系方式用于错误勘正

## Overview / 概述

This Jupyter Notebook demonstrates a BERT-based approach for text classification, where we apply BERT to classify text data. The notebook covers the entire process, from loading and preprocessing data, to tokenization, model training, and evaluation.
本 Jupyter 笔记本演示了基于 BERT 的文本分类方法，利用 BERT 进行文本数据的分类。该笔记本涵盖了从数据加载与预处理、分词、模型训练到评估的完整过程。

## Libraries & Tools Used / 使用的库与工具

- **BERT (Bidirectional Encoder Representations from Transformers)**: A transformer-based model pre-trained on a large corpus of text and fine-tuned for specific NLP tasks.
  **BERT（双向编码器表示的变换器）**：一种基于变换器的模型，预先训练于大规模文本语料库，并针对特定的 NLP 任务进行微调。
- **PyTorch**: A deep learning framework used for model training and optimization.
  **PyTorch**：用于模型训练和优化的深度学习框架。
- **HuggingFace Transformers**: A library for state-of-the-art NLP models, providing easy access to BERT.
  **HuggingFace Transformers**：一个用于最新 NLP 模型的库，提供对 BERT 等模型的便捷访问。
- **Scikit-learn**: Used for evaluation metrics and data handling.
  **Scikit-learn**：用于评估指标和数据处理。

## Steps Covered in the Notebook / 笔记本中涵盖的步骤

1. **Data Loading and Preprocessing / 数据加载与预处理**:

   - Loading data from `.txt` files, which contain text data (questions) and corresponding labels.
     从`.txt`文件加载数据，其中包含文本数据（问题）和相应的标签。
   - Tokenizing the text using BERT's tokenizer.
     使用 BERT 的分词器对文本进行分词。

2. **Model Setup / 模型设置**:

   - Defining the BERT-based model, including various layers for feature extraction and classification.
     定义基于 BERT 的模型，包括用于特征提取和分类的各种层。
   - Implementing a custom attention mechanism and LSTM layers.
     实现自定义的注意力机制和 LSTM 层。

3. **Model Training / 模型训练**:

   - Training the BERT-based model on the dataset.
     在数据集上训练基于 BERT 的模型。
   - Experimenting with techniques such as adversarial training (FGM) and R-Drop regularization.
     尝试使用对抗训练（FGM）和 R-Drop 正则化等技术。

4. **Evaluation / 评估**:

   - Evaluating the model on a validation set using accuracy and other metrics.
     使用准确率和其他评估指标在验证集上评估模型。

5. **Prediction and Submission / 预测与提交**:
   - Making predictions on the test set and preparing the results for submission.
     在测试集上进行预测并准备提交结果。

## 关于题目

- 天池大数据竞赛 NCAA2024-中文糖尿病问题分类 notebook
- 数据来源，课程自用

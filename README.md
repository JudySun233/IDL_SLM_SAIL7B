# IDL_SLM_SAIL7B
11785 Spring 2025 Project

## Project Structure

```
├── IDL_SLM_SAIL7B/
│   ├── baselines/
│   │   ├── climate-fever/
│   │   │   ├── LLAMA7B_climate.ipynb
│   │   │   ├── LLAMA7B-climate.log
│   │   │   ├── SAIL7B_climate.ipynb
│   │   │   └── SAIL7B-climate.log
│   │   ├── hate-speech-detection/
│   │   │   ├── Llama_hate_speech_results_gpt_gt.csv
│   │   │   ├── llama_test_gpt_gt.ipynb
│   │   │   ├── llama_test_gt.ipynb
│   │   │   ├── SAIL-7B-hate_speech_gptmini4o_ground_truth.log
│   │   │   ├── SAIL-7B-hate_speech_labels_ground_truth.log
│   │   │   └── SAIL-7B-hate_speech.ipynb
│   ├── README.md
```

## Datasets

The evaluation uses two benchmark datasets:
1. **Climate-Fever** (Diggelmann et al., 2020): Fact-checking dataset with climate change claims labeled as "Supports or, "Refutes"
2. **Hate Speech Detection (HSD)** (de Gibert et al., 2018): Dataset containing online discussions categorized as hate speech or non-hate speech.

## Methodology

The experiments follow a zero-shot inference approach:
1. Dataset preparation and filtering
2. Model and tokenizer setup using PyTorch
3. Binary classification prompts without reasoning explanations
4. Inference pipeline using GPT-4o-mini for ground truth generation
5. Performance evaluation using accuracy and F1 score metrics

## Models:
1. Finetuned SAIL7B model with deberta dataset: https://drive.google.com/drive/folders/1XIuVksnJXRNewCcNNx8pIe1dv82Kq2-M?usp=sharing
2. Finetuned SAIL7B model with mistral dataset: https://drive.google.com/drive/folders/1hQJ78oXWjmPPXkKaUR7c1ygD8X2mMQVt?usp=sharing
   

## Results

Overall, SAIL-7B outperforms LLaMA-7B across both tasks, showing stronger fact-checking capabilities. The Climate-Fever results align with the original SAIL paper, while the HSD results show some discrepancies from the reported metrics.

## Usage
The notebooks in each subdirectory contain the complete code for reproducing the experiments with each model on the respective datasets.

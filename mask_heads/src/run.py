import gzip
import json
import math
import os

import huggingface_hub
import hydra
import pandas as pd
import torch
import wandb
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.configs import RunnerConfigs
from src.factories import get_dataset, get_metrics, get_model


class Run:
    def __init__(self, configs: RunnerConfigs):
        self.configs = configs

        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]

        self._load_dataloaders()
        self._load_pipeline()
        self._load_metrics()

        self._setup_run()

    def _load_dataloaders(self) -> None:
        self.dataloaders = {}

        self.dataset = get_dataset(
            self.configs.data,
            use_chat_template=self.configs.model.model_type == "instruct",
        )
        self.dataloaders = DataLoader(
            self.dataset,
            shuffle=False,
            **self.configs.data_loader,
        )

    def _load_pipeline(self) -> None:
        self.model = get_model(self.configs.model, self.configs.decoder)

    def _load_metrics(self):
        self.metrics = get_metrics(self.configs.data)

    def _setup_run(self):
        # Naming by model name
        self.run_name = f"{self.configs.model.name}__{self.configs.decoder.name}"
        # print(self.configs.debug)
        # if not self.configs.debug:
        #     self.group_name = self.configs.data.name
        #     wandb.init(
        #         project=self.configs.wandb_project,
        #         entity=self.configs.wandb_entity,
        #         name=self.run_name,
        #         group=self.group_name,
        #         config=OmegaConf.to_container(self.configs),
        #     )

    def test(self):
        """
        Test the model on the dataset and log the predictions and metrics to WandB
        """
        predictions = []

        prediction_filename = f"pred_{self.configs.data.name}_{self.run_name}"

        answers_filepath = os.path.join(
            self.output_dir, f"answer_{prediction_filename}.npy"
        )
        scores_filepath = os.path.join(
            self.output_dir, f"score_{self.configs.data.name}.npy"
        )

        # To save WandB space, just return attentions for the Baseline model
        # Mainly for Logistic Regression purposes
        self.answers = [None]*len(self.dataloaders)
        self.scores = [None]*len(self.dataloaders)
        for step, batch in enumerate(tqdm(self.dataloaders)):
            # Predict
            prediction = self.model.generate(batch)
            self.answers[step] = prediction["decoded_text"]
            self.scores[step] = prediction["score"]
        np.save(answers_filepath, self.answers)
        np.save(scores_filepath, self.scores)

    def evaluate(self):
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        model = BleurtForSequenceClassification.from_pretrained('/data/TAP/wangyujing/decore/BLEURT-20').cuda()
        tokenizer = BleurtTokenizer.from_pretrained('/data/TAP/wangyujing/decore/BLEURT-20')
        model.eval()

        dataset = self.dataset
        gts = np.zeros(0)
        length = len(dataset)
        prediction_filename = f"pred_{self.configs.data.name}_{self.run_name}"
        answers_filepath = os.path.join(self.output_dir, f"answer_{prediction_filename}.npy")
        all_predictions = np.load(answers_filepath)
        # all_predictions = np.load(f'/data/TAP/wangyujing/decore/outputs/2025-07-13/01-13-43/answer_pred_TruthfulQA_LLaMA3-8b-Instruct__DeCoReVanilla.npy') #self.answers
            
        for i in range(length):
            
            best_answer = dataset[i]['answer_best']
            correct_answer = dataset[i]['answer_true'].split('; ')
            incorrect_answer = dataset[i]['answer_false'].split('; ')
            all_answers = [best_answer] + correct_answer
            question = dataset[i]['question']
                
            # get the gt.
            predictions = [all_predictions[i]]
            # print(predictions)
            all_results_correct = np.zeros((len(all_answers), len(predictions)))
            all_results_incorrect = np.zeros((len(incorrect_answer), len(predictions)))
            with torch.no_grad():
                for anw in range(len(all_answers)):
                    inputs = tokenizer(predictions, [all_answers[anw]] * len(predictions),
                                        padding='longest', return_tensors='pt')
                    for key in list(inputs.keys()):
                        inputs[key] = inputs[key].cuda()
                    res_correct = np.asarray(model(**inputs).logits.flatten().tolist())
                    all_results_correct[anw] = res_correct
                for anw in range(len(incorrect_answer)):
                    inputs = tokenizer(predictions, [incorrect_answer[anw]] * len(predictions),
                                        padding='longest', return_tensors='pt')
                    for key in list(inputs.keys()):
                        inputs[key] = inputs[key].cuda()
                    res_incorrect = np.asarray(model(**inputs).logits.flatten().tolist())
                    all_results_incorrect[anw] = res_incorrect
            # gts = np.concatenate([gts, np.max(all_results_correct, axis=0)], 0)
            if np.max(all_results_correct, axis=0) > np.max(all_results_incorrect, axis=0):
                gts = np.concatenate([gts, np.max(all_results_correct, axis=0)], 0)
            else:
                gts = np.concatenate([gts, np.array([0.0])], 0)
            if i % 10 == 0:
                print("samples passed: ", i)

        np.save(os.path.join(
            self.output_dir,f'./gts_{self.configs.data.name}_bleurt_score.npy'), gts)
        
        from sklearn.metrics import roc_auc_score, accuracy_score
        # gts = np.load('/data/TAP/wangyujing/decore/outputs/2025-07-12/23-44-18/gts_TruthfulQA_bleurt_score.npy')
        gts = np.load(os.path.join(
            self.output_dir,f'./gts_{self.configs.data.name}_bleurt_score.npy'))
        scores_filepath = os.path.join(
            self.output_dir, f"score_{self.configs.data.name}.npy"
        )
        scores = np.load(scores_filepath)
        # scores = np.load(f'/data/TAP/wangyujing/decore/outputs/2025-07-13/01-13-43/score_TruthfulQA.npy')
        gt_label = np.asarray(gts> 0.5, dtype=np.int32)
        # print(scores)
        # print(gts)
        # print(gt_label)
        # scores[scores < 0.55] = 0.0
        all = 0
        false = 0
        for (score, gt) in zip(scores, gt_label):
            if score<=0.5:
                all += 1
                if gt==1:
                    false += 1
        print(f"false: {false}, all: {all}")
        # scaled_scores = (scores - scores.min()) / (scores.max() - scores.min())
        score_res = np.asarray(scores> 0.5, dtype=np.int32)
        test_auroc = roc_auc_score(gt_label, scores)
        # score_res = np.asarray(scores> 0.65, dtype=np.int32)
        test_acc = accuracy_score(gt_label, score_res)
        print(f"test auroc: {test_auroc}")
        print(f"test acc: {test_acc}")

    def post_analyze(self):
        gts = list(np.load('/data/TAP/wangyujing/decore/outputs/2025-07-13/01-13-43/gts_TruthfulQA_bleurt_score.npy'))
        score = list(np.load('/data/TAP/wangyujing/decore/outputs/2025-07-13/01-13-43/score_TruthfulQA.npy'))
        # answers = list(np.load('/data/TAP/wangyujing/decore/outputs/2025-07-07/20-23-46/answer_pred_TruthfulQA_LLaMA3-8b-Instruct__DeCoReVanilla.npy'))
        # for i in range(len(score)):
        #     if gts[i] == 0 and score[i] < 0.5:
        #         print(f'{i} th answer: {answers[i]}')
        factual_similarities = []
        hallucinated_similarities = []
        for i in range(len(score)):
            if gts[i] > 0.5:
                factual_similarities.append(score[i])
            else:
                hallucinated_similarities.append(score[i])
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 5))

        sns.kdeplot(factual_similarities, label='Factual', shade=True, color='green', linewidth=2)
        sns.kdeplot(hallucinated_similarities, label='Hallucinated', shade=True, color='red', linewidth=2)

        plt.xlabel("Cosine Similarity", fontsize=16)
        plt.ylabel("Density", fontsize=16)
        # plt.title("Distribution of Cosine Similarity")
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        # 保存图像
        save_path = "/data/TAP/wangyujing/decore/outputs/2025-07-13/01-13-43/cosine_similarity_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 300 dpi for high quality
        plt.close()
            # print(step)
            # if step==50:
            #     break
            # batch["predicted_answer"] = prediction["decoded_text"]
            # if "alphas" in prediction:
            #     # Handle for DeCoRe guided, to analyse the changes in alpha value throughout generation steps
            #     batch["alphas"] = prediction["alphas"]

            # if self.configs.data.name in ["TruthfulQA", "MemoTrap"]:
            #     scores_true = []
            #     scores_false = []
            #     for temp_ans in batch["prompted_ref_true"]:
            #         ans = temp_ans[0] if type(temp_ans) in [list, tuple] else temp_ans
            #         log_probs = self.model.lm_score(batch, ans)
            #         scores_true.append(log_probs)

            #     for temp_ans in batch["prompted_ref_false"]:
            #         ans = temp_ans[0] if type(temp_ans) in [list, tuple] else temp_ans
            #         log_probs = self.model.lm_score(batch, ans)
            #         scores_false.append(log_probs)

            #     batch["scores_true"] = scores_true
            #     batch["scores_false"] = scores_false

            # if self.configs.data.name == "MemoTrap":
            #     batch["answer_index"] = int(batch["answer_index"].cpu().numpy()[0])

            # # Brute force normalisation for IFEval, some values were casted as tensors by collator
            # if self.configs.data.name == "IFEval":
            #     batch["kwargs"] = [
            #         {
            #             k: int(v.cpu().numpy()[0]) if type(v) == torch.Tensor else v
            #             for k, v in kwargs_.items()
            #         }
            #         for kwargs_ in batch["kwargs"]
            #     ]

            # predictions.append(batch)

            # values_to_normalised = ["idx"]
            # if self.configs.data.name == "PopQA":
            #     values_to_normalised += [
            #         "s_pop",
            #         "o_pop",
            #     ]
            # for key in values_to_normalised:
            #     try:
            #         batch[key] = int(batch[key].cpu().numpy()[0])
            #     except:
            #         batch[key] = str(batch[key][0])

            # # Save the predictions to a JSONL file after each batch
            # with open(prediction_filepath, "a") as f:
            #     f.write(json.dumps(batch) + "\n")

        # # Evaluate
        # metrics = self.metrics(predictions)
        # print(metrics)

        # Log
        # if not self.configs.debug:
        #     wandb.log(metrics)

        #     pred_artifact = wandb.Artifact(prediction_filename, type="prediction")
        #     pred_artifact.add_file(prediction_filepath)
        #     wandb.log_artifact(pred_artifact)
        # else:
        #     print(metrics)

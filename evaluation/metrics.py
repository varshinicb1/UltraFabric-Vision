from sklearn.metrics import roc_auc_score, f1_score
import time

class EvaluationEngine:
    def __init__(self):
        self.results = {}

    def evaluate_model(self, model, dataloader, ground_truth):
        """
        Evaluate a given model.
        ground_truth: list of tuples (label, mask)
        """
        y_true = []
        y_pred = []
        inference_times = []
        
        for i, x in enumerate(dataloader):
            x = x.to(model.device)
            start_time = time.time()
            score, heatmap = model.predict(x)
            inference_times.append(time.time() - start_time)
            
            y_pred.append(score)
            y_true.append(ground_truth[i][0])
            
        auroc = roc_auc_score(y_true, y_pred)
        # Using a threshold of 0.5 for F1 (typically you'd optimize this)
        preds_binary = [1 if p > 0.5 else 0 for p in y_pred]
        f1 = f1_score(y_true, preds_binary)
        
        avg_time = sum(inference_times) / len(inference_times)
        
        return {
            "auroc": auroc,
            "f1_score": f1,
            "avg_inference_time": avg_time
        }

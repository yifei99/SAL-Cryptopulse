import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from scipy.signal import correlate
from sklearn import metrics

from cryptopulse.workflow import Workflow
from cryptopulse.data import data_provider, DatasetStandard

# 获取项目根目录
ROOT_DIR = Path(__file__).parent

def calculate_metrics(pred, true):
    """计算评估指标，与训练时保持一致"""
    mae = metrics.mean_absolute_error(true, pred)
    mse = metrics.mean_squared_error(true, pred)
    corr = correlate(pred, true)
    corr = corr[len(corr) // 2]
    corr = corr / np.sqrt(np.sum(pred**2) * np.sum(true**2))
    pearson_corr = np.corrcoef(pred, true)[0, 1]
    return mae, mse, corr, pearson_corr

def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='加密货币价格预测评估')
    parser.add_argument("--ob-len", type=int, default=7, help="观察序列长度（天数）")
    parser.add_argument("--pred-len", type=int, default=1, help="预测序列长度（天数）")
    parser.add_argument(
        "--data", 
        type=str, 
        default="BTC-USD",
        help="要预测的加密货币"
    )
    parser.add_argument("--d-model", type=int, default=512, help="模型维度")
    parser.add_argument("--n-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--exp-name", type=str, default="Test", help="实验名称")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    args = parser.parse_args()

    # 初始化工作流
    workflow = Workflow(args)
    
    # 加载模型
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{checkpoint_path}")
    
    try:
        workflow.model.load_state_dict(torch.load(checkpoint_path))
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        return

    # 获取测试数据加载器和数据集
    test_loader = workflow.get_data("test")
    test_dataset = test_loader.dataset
    
    # 存储预测结果
    results = []
    all_preds_scaled = []
    all_trues_scaled = []
    
    # 在测试集上进行预测
    workflow.model.eval()
    with torch.no_grad():
        for batch_idx, (x_data, y_data) in enumerate(tqdm(test_loader, desc="生成预测")):
            # 将数据移到正确的设备上
            x_data = x_data.float().to(workflow.device)
            y_data = y_data.float().to(workflow.device)
            
            # 预测
            preds = workflow.model(x_data)
            preds = preds[:, -args.pred_len:, -1:]  # 只取最后一个特征（价格）
            trues = y_data[:, -args.pred_len:, -1:]
            
            # 转换为 numpy 数组
            preds = preds.cpu().numpy()
            trues = trues.cpu().numpy()
            
            # 保存标准化空间的值用于计算相关系数
            all_preds_scaled.extend(preds[:, -1, -1])
            all_trues_scaled.extend(trues[:, -1, -1])
            
            # 记录每个批次中的预测结果
            for i in range(preds.shape[0]):
                # 反标准化预测值和实际值
                pred_price = test_dataset.scaler.inverse_transform(
                    np.array([[0] * (test_dataset.data.shape[1] - 1) + [preds[i, -1, -1]]])
                )[0, -1]
                
                true_price = test_dataset.scaler.inverse_transform(
                    np.array([[0] * (test_dataset.data.shape[1] - 1) + [trues[i, -1, -1]]])
                )[0, -1]
                
                results.append({
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "predicted_price": pred_price,
                    "actual_price": true_price,
                })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    results_df["error"] = results_df["predicted_price"] - results_df["actual_price"]
    results_df["error_percent"] = (results_df["error"] / results_df["actual_price"]) * 100
    
    # 创建输出目录
    output_dir = ROOT_DIR / "evaluations"
    output_dir.mkdir(exist_ok=True)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.data}_eval_{timestamp}.csv"
    
    # 保存结果
    results_df.to_csv(output_file, index=False)
    print(f"\n评估结果已保存到：{output_file}")
    
    # 计算标准化空间的指标
    mae_scaled, mse_scaled, corr_scaled, pearson_corr_scaled = calculate_metrics(
        np.array(all_preds_scaled), 
        np.array(all_trues_scaled)
    )
    
    # 计算实际价格空间的指标
    mae_raw, mse_raw, corr_raw, pearson_corr_raw = calculate_metrics(
        results_df["predicted_price"].values,
        results_df["actual_price"].values
    )
    
    # 打印统计信息
    print("\n标准化空间的评估统计：")
    print(f"总样本数：{len(results_df)}")
    print(f"平均绝对误差 (MAE): {mae_scaled:.4f}")
    print(f"均方误差 (MSE): {mse_scaled:.4f}")
    print(f"自相关系数: {corr_scaled:.4f}")
    print(f"皮尔逊相关系数: {pearson_corr_scaled:.4f}")
    
    print("\n实际价格空间的评估统计：")
    print(f"平均绝对误差 (MAE): ${mae_raw:.2f}")
    print(f"均方误差 (MSE): ${mse_raw:.2f}")
    print(f"自相关系数: {corr_raw:.4f}")
    print(f"皮尔逊相关系数: {pearson_corr_raw:.4f}")
    print(f"平均百分比误差: {results_df['error_percent'].mean():.2f}%")
    print(f"预测准确方向的比例: {(results_df['error_percent'] * results_df['error_percent'].shift(1) > 0).mean():.2%}")
    
    # 打印价格范围
    print("\n价格范围：")
    print(f"预测价格范围：${results_df['predicted_price'].min():.2f} - ${results_df['predicted_price'].max():.2f}")
    print(f"实际价格范围：${results_df['actual_price'].min():.2f} - ${results_df['actual_price'].max():.2f}")

if __name__ == "__main__":
    main() 
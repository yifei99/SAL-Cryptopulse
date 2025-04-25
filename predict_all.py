import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from scipy.signal import correlate
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from cryptopulse.workflow import Workflow
from cryptopulse.data import split_data

# 获取项目根目录
ROOT_DIR = Path(__file__).parent
DATASET_DIR = ROOT_DIR / "dataset"

# 可用的加密货币列表
AVAILABLE_CRYPTOS = {
    "BTC-USD": "1.BTC-USD.csv",
    "ETH-USD": "2.ETH-USD.csv",
    "USDT-USD": "3.USDT-USD.csv",
    "BNB-USD": "4.BNB-USD.csv",
    "SOL-USD": "5.SOL-USD.csv",
    "STETH-USD": "6.STETH-USD.csv",
    "USDC-USD": "7.USDC-USD.csv",
    "XRP-USD": "8.XRP-USD.csv",
    "DOGE-USD": "9.DOGE-USD.csv",
    "TON11419-USD": "10.TON11419-USD.csv",
    "ADA-USD": "11.ADA-USD.csv",
    "SHIB-USD": "12.SHIB-USD.csv",
    "AVAX-USD": "13.AVAX-USD.csv",
    "WSTETH-USD": "14.WSTETH-USD.csv",
    "WETH-USD": "15.WETH-USD.csv",
    "DOT-USD": "16.DOT-USD.csv",
    "LINK-USD": "17.LINK-USD.csv",
    "WBTC-USD": "18.WBTC-USD.csv",
    "TRX-USD": "19.TRX-USD.csv",
    "WTRX-USD": "20.WTRX-USD.csv"
}

def calculate_metrics(pred, true):
    """计算评估指标"""
    mae = metrics.mean_absolute_error(true, pred)
    mse = metrics.mean_squared_error(true, pred)
    corr = correlate(pred, true)
    corr = corr[len(corr) // 2]
    corr = corr / np.sqrt(np.sum(pred**2) * np.sum(true**2))
    pearson_corr = np.corrcoef(pred, true)[0, 1]
    return mae, mse, corr, pearson_corr

def prepare_data(df_raw, ob_len):
    """准备预测所需的数据"""
    # 准备特征数据
    feature_cols = [col for col in df_raw.columns if col not in ["date", "close"]] + ["close"]
    df_data = df_raw[feature_cols]
    
    # 标准化数据
    scaler = StandardScaler()
    scaler.fit(df_data)
    data_scaled = scaler.transform(df_data)
    
    return data_scaled, scaler, feature_cols

def get_observation(data_scaled, start_idx, ob_len):
    """获取指定位置的观察序列"""
    if start_idx < 0:
        padding = np.zeros((abs(start_idx), data_scaled.shape[1]))
        seq_data = np.vstack([padding, data_scaled[:start_idx + ob_len]])
    else:
        seq_data = data_scaled[start_idx:start_idx + ob_len]
    return seq_data

def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='加密货币价格全历史预测')
    parser.add_argument("--ob-len", type=int, default=7, help="观察序列长度（天数）")
    parser.add_argument("--pred-len", type=int, default=1, help="预测序列长度（天数）")
    parser.add_argument(
        "--data", 
        type=str, 
        default="BTC-USD",
        choices=list(AVAILABLE_CRYPTOS.keys()),
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

    # 读取数据
    data_path = DATASET_DIR / AVAILABLE_CRYPTOS[args.data]
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")
    
    df_raw = pd.read_csv(data_path)
    df_raw.columns = df_raw.columns.str.lower()
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values(by=["date"])
    
    # 准备数据
    data_scaled, scaler, feature_cols = prepare_data(df_raw, args.ob_len)
    
    # 存储预测结果
    results = []
    all_preds_scaled = []
    all_trues_scaled = []
    
    # 在所有数据上进行预测
    workflow.model.eval()
    with torch.no_grad():
        # 从第ob_len天开始预测
        for i in tqdm(range(args.ob_len, len(df_raw)), desc="生成预测"):
            # 获取观察序列
            x_data = get_observation(data_scaled, i-args.ob_len, args.ob_len)
            x_data = torch.FloatTensor(x_data).unsqueeze(0)
            
            # 预测
            pred = workflow.model(x_data.to(workflow.device))
            pred = pred[0, -1, -1].cpu().item()  # 只取最后一个时间步的价格预测
            
            # 获取实际值
            true = data_scaled[i, -1]  # 最后一列是价格
            
            # 保存标准化空间的值
            all_preds_scaled.append(pred)
            all_trues_scaled.append(true)
            
            # 反标准化预测值和实际值
            dummy_data = np.zeros((1, len(feature_cols)))
            
            dummy_data[0, -1] = pred
            pred_price = scaler.inverse_transform(dummy_data)[0, -1]
            
            dummy_data[0, -1] = true
            true_price = scaler.inverse_transform(dummy_data)[0, -1]
            
            # 获取前一天的价格
            prev_price = df_raw.iloc[i-1]["close"]
            
            # 记录结果
            results.append({
                "date": df_raw.iloc[i]["date"],
                "prev_price": prev_price,
                "predicted_price": pred_price,
                "actual_price": true_price,
                "predicted_change": pred_price - prev_price,
                "actual_change": true_price - prev_price,
                "predicted_change_percent": ((pred_price - prev_price) / prev_price) * 100,
                "actual_change_percent": ((true_price - prev_price) / prev_price) * 100
            })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    results_df["error"] = results_df["predicted_price"] - results_df["actual_price"]
    results_df["error_percent"] = (results_df["error"] / results_df["actual_price"]) * 100
    
    # 创建输出目录
    output_dir = ROOT_DIR / "predictions"
    output_dir.mkdir(exist_ok=True)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.data}_all_predictions_{timestamp}.csv"
    
    # 保存结果
    results_df.to_csv(output_file, index=False)
    print(f"\n预测结果已保存到：{output_file}")
    
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
    
    # 计算按年份的统计
    print("\n按年份的统计：")
    results_df['year'] = pd.to_datetime(results_df['date']).dt.year
    yearly_stats = results_df.groupby('year').agg({
        'error_percent': ['mean', 'std'],
        'predicted_change_percent': ['mean', 'std'],
        'actual_change_percent': ['mean', 'std'],
        'predicted_price': ['min', 'max'],
        'actual_price': ['min', 'max']
    }).round(2)
    print(yearly_stats)
    
    # 打印整体价格范围
    print("\n整体价格范围：")
    print(f"预测价格范围：${results_df['predicted_price'].min():.2f} - ${results_df['predicted_price'].max():.2f}")
    print(f"实际价格范围：${results_df['actual_price'].min():.2f} - ${results_df['actual_price'].max():.2f}")
    
    # 打印变化范围
    print("\n价格变化范围：")
    print(f"预测变化范围：${results_df['predicted_change'].min():.2f} - ${results_df['predicted_change'].max():.2f}")
    print(f"实际变化范围：${results_df['actual_change'].min():.2f} - ${results_df['actual_change'].max():.2f}")
    print(f"预测变化率范围：{results_df['predicted_change_percent'].min():.2f}% - {results_df['predicted_change_percent'].max():.2f}%")
    print(f"实际变化率范围：{results_df['actual_change_percent'].min():.2f}% - {results_df['actual_change_percent'].max():.2f}%")

if __name__ == "__main__":
    main() 
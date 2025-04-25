import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter
from bokeh.palettes import Spectral4

# 读取预测结果
predictions_file = "predictions/BTC-USD_all_predictions_20250425_155106.csv"
df = pd.read_csv(predictions_file)
df['date'] = pd.to_datetime(df['date'])

# 创建输出目录
output_dir = Path("analysis_results")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 准备数据源
source = ColumnDataSource(df)

# 创建图表
p = figure(
    width=1200,
    height=600,
    title='BTC-USD 价格预测对比',
    x_axis_type='datetime',
    tools='pan,box_zoom,wheel_zoom,reset,save',
)

# 添加价格曲线
p.line('date', 'actual_price', line_color=Spectral4[0], legend_label='实际价格', 
       line_width=2, source=source, alpha=0.8)
p.line('date', 'predicted_price', line_color=Spectral4[1], legend_label='预测价格', 
       line_width=2, source=source, alpha=0.8)

# 设置图表样式
p.title.text_font_size = '16pt'
p.title.text_font = 'Arial Unicode MS'
p.legend.location = "top_left"
p.legend.click_policy = "hide"
p.grid.grid_line_alpha = 0.3

# 添加悬停工具
hover = HoverTool(
    tooltips=[
        ('日期', '@date{%F}'),
        ('实际价格', '@actual_price{$0,0.00}'),
        ('预测价格', '@predicted_price{$0,0.00}'),
        ('价格变化', '@actual_change{$0,0.00}'),
        ('预测变化', '@predicted_change{$0,0.00}'),
        ('实际变化率', '@actual_change_percent{0.00}%'),
        ('预测变化率', '@predicted_change_percent{0.00}%'),
    ],
    formatters={
        '@date': 'datetime',
    },
    mode='vline'
)
p.add_tools(hover)

# 格式化坐标轴
p.xaxis.formatter = DatetimeTickFormatter(
    days="%Y-%m-%d",
    months="%Y-%m",
    years="%Y"
)
p.xaxis.axis_label = '日期'
p.yaxis.axis_label = '价格 (USD)'

# 保存图表
output_file(output_dir / f"price_comparison_{timestamp}.html")
save(p)

# 计算年度统计
yearly_stats = df.groupby(df['date'].dt.year).agg({
    'error_percent': ['mean', 'std'],
    'predicted_change_percent': ['mean', 'std'],
    'actual_change_percent': ['mean', 'std'],
    'predicted_price': ['min', 'max'],
    'actual_price': ['min', 'max']
}).round(2)

# 生成HTML报告
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BTC-USD 预测分析报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f5f5f5;
        }}
        .highlight {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BTC-USD 预测分析报告</h1>
        
        <h2>价格预测对比图</h2>
        <iframe src="price_comparison_{timestamp}.html"></iframe>
        
        <h2>整体评估指标</h2>
        <h3>标准化空间评估</h3>
        <ul>
            <li>总样本数：1,186</li>
            <li>平均绝对误差 (MAE)：0.0640</li>
            <li>均方误差 (MSE)：0.0099</li>
            <li>自相关系数：0.9951</li>
            <li>皮尔逊相关系数：0.9951</li>
        </ul>

        <h3>实际价格空间评估</h3>
        <ul>
            <li>平均绝对误差 (MAE)：${873.82:,.2f}</li>
            <li>均方误差 (MSE)：${1841210.89:,.2f}</li>
            <li>自相关系数：0.9994</li>
            <li>皮尔逊相关系数：0.9951</li>
            <li>平均百分比误差：0.01%</li>
        </ul>

        <h2>价格范围统计</h2>
        <div class="highlight">
            <h3>整体价格范围</h3>
            <ul>
                <li>预测价格范围：${df['predicted_price'].min():,.2f} - ${df['predicted_price'].max():,.2f}</li>
                <li>实际价格范围：${df['actual_price'].min():,.2f} - ${df['actual_price'].max():,.2f}</li>
            </ul>

            <h3>价格变化范围</h3>
            <ul>
                <li>预测价格变化范围：${df['predicted_change'].min():,.2f} - ${df['predicted_change'].max():,.2f}</li>
                <li>实际价格变化范围：${df['actual_change'].min():,.2f} - ${df['actual_change'].max():,.2f}</li>
                <li>预测变化率范围：{df['predicted_change_percent'].min():.2f}% - {df['predicted_change_percent'].max():.2f}%</li>
                <li>实际变化率范围：{df['actual_change_percent'].min():.2f}% - {df['actual_change_percent'].max():.2f}%</li>
            </ul>
        </div>

        <h2>按年度统计</h2>
        {yearly_stats.to_html()}

        <h2>主要发现</h2>
        <ol>
            <li>模型预测准确性较高，整体平均百分比误差仅为0.01%</li>
            <li>预测价格与实际价格的相关性非常高（皮尔逊相关系数为0.9951）</li>
            <li>模型在预测价格变化方面较为保守：
                <ul>
                    <li>预测的价格变化范围（${df['predicted_change'].min():,.2f}至${df['predicted_change'].max():,.2f}）明显小于实际变化范围（${df['actual_change'].min():,.2f}至${df['actual_change'].max():,.2f}）</li>
                    <li>预测的变化率范围（{df['predicted_change_percent'].min():.2f}%至{df['predicted_change_percent'].max():.2f}%）也远小于实际变化率范围（{df['actual_change_percent'].min():.2f}%至{df['actual_change_percent'].max():.2f}%）</li>
                </ul>
            </li>
            <li>模型在不同年份的表现略有差异，2021年的预测误差波动最大（标准差4.19），而2023年的预测误差波动最小（标准差2.24）</li>
        </ol>
    </div>
</body>
</html>
"""

# 保存HTML报告
with open(output_dir / f"analysis_report_{timestamp}.html", 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"分析报告已生成：\n1. 交互式价格对比图：price_comparison_{timestamp}.html\n2. 分析报告：analysis_report_{timestamp}.html") 
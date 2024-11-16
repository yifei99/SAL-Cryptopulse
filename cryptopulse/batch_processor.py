import os
import subprocess
import sys

def batch_process():
    # Set PYTHONPATH to current directory so Python recognizes `cryptopulse`
    os.environ['PYTHONPATH'] = os.getcwd()

    ob_len = 7
    pred_len = 1
    batch_size = 8
    top20 = [
        "BTC", "ETH", "USDT", "BNB", "SOL", "STETH", "USDC", "XRP", "DOGE", "TON11419",
        "ADA", "SHIB", "AVAX", "WSTETH", "WETH", "DOT", "LINK", "WBTC", "TRX", "WTRX"
    ]
    exp_name = "cryptopulse_results"
    log_folder = f"./logs/{exp_name}"

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Get the path to the python executable in the virtual environment
    python_executable = sys.executable

    for data in top20:
        data += "-USD"
        filename = f"{log_folder}/{data}.log"
        print(f"Processing {data}... Results will be in 'results/{exp_name}'")
        
        # Running the cryptopulse.main with required arguments, use the python executable from the virtual environment
        command = [
            python_executable, '-u', '-m', 'cryptopulse.main',
            '--data', data,
            '--use-cuda',
            '--exp-name', exp_name,
            '--ob-len', str(ob_len),
            '--pred-len', str(pred_len),
            '--batch-size', str(batch_size),
            '--learning-rate', '0.0005'
        ]
        with open(filename, 'w') as log_file:
            subprocess.run(command, stdout=log_file)

    print("Script completed.")

if __name__ == "__main__":
    batch_process()

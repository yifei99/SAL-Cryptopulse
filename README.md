# SAL-CryptoPulse

The official implementation of the paper "CryptoPulse: Short-Term Cryptocurrency Forecasting with Dual-Prediction and Cross-Correlated Market Indicators". Portions of the code have been adapted from the DLinear implementation.

![model](https://github.com/user-attachments/assets/e09ad3c1-ba77-4f7a-b9da-96ac04a5e1aa)

## 🏆 Top 20 Cryptocurrencies
Below is the list of the top 20 cryptocurrencies (by market capitalization at the time of data collection), along with the exact CSV file names (which include the -USD ticker format). The numeric prefix in each CSV file indicates its rank. You can see these file names in the dataset folder:

BTC-USD (1.BTC-USD.csv)

ETH-USD (2.ETH-USD.csv)

USDT-USD (3.USDT-USD.csv)

BNB-USD (4.BNB-USD.csv)

SOL-USD (5.SOL-USD.csv)

STETH-USD (6.STETH-USD.csv)

XRP-USD (7.XRP-USD.csv)

USDC-USD (8.USDC-USD.csv)

DOGE-USD (9.DOGE-USD.csv)

TON11419-USD (10.TON11419-USD.csv)

ADA-USD (11.ADA-USD.csv)

SHIB-USD (12.SHIB-USD.csv)

AVAX-USD (13.AVAX-USD.csv)

WSTETH-USD (14.WSTETH-USD.csv)

WETH-USD (15.WETH-USD.csv)

DOT-USD (16.DOT-USD.csv)

LINK-USD (17.LINK-USD.csv)

WBTC-USD (18.WBTC-USD.csv)

TRX-USD (19.TRX-USD.csv)

WTRX-USD (20.WTRX-USD.csv)


## 🚀 Features


- **Short-Term Forecasting**: Predict cryptocurrency prices with a model designed for short-term predictions(next-day).
- **Batch Processing**: Run predictions for multiple cryptocurrencies at once.
- **Flexible & Easy to Use**: Install, run, and watch the results roll in!

## ⚙️ Installation

You can install the package directly from GitHub with `pip`:

        pip install git+https://github.com/aamitssharma07/SAL-Cryptopulse.git

Alternatively, you can install it manually:

1. Clone the repo:

        git clone https://github.com/aamitssharma07/SAL-Cryptopulse.git

        cd SAL-Cryptopulse

2. Install dependencies in a virtual environment:

        python -m venv env_crypto

        env_crypto\Scripts\activate

        pip install -e .

## 🎯 Usage

After installation, you can run the model directly using the package's command line interface if you had installed the package directly from GitHub with `pip :

    For a single run:
    
        cryptopulse --data <crypto-ticker-symbol> --train-epochs 10 --batch-size 32 

    For batch processing (multiple cryptos at once):
    
        cryptopulse_batch

Alternatively, you can run the model using the Python command if you have cloned the repo in your local

    For a single run:
    
        python -m cryptopulse.main --data BTC-USD --train-epochs 10 --batch-size 32

    For batch processing (multiple cryptos at once):
    
        python -m cryptopulse.batch_processor

## 📊 Results

All results are saved in the results/cryptopulse_results directory.

## 💡 Contributing

Feel free to use this research work. Let’s build a smarter CryptoPulse together!
You can connect with me on LinkedIn: https://www.linkedin.com/in/aamit-datascientist/

## 📝 Citation
@article{kumar2025cryptopulse,
  title={CryptoPulse: Short-Term Cryptocurrency Forecasting with Dual-Prediction and Cross-Correlated Market Indicators},
  author={Kumar, Amit and Ji, Taoran},
  journal={arXiv preprint arXiv:2502.19349},
  year={2025}
}

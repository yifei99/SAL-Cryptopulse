#! /bin/bash

ob_len=7
pred_len=1
batch_size=8
top20=("BTC" "ETH" "USDT" "BNB" "SOL" "STETH" "USDC" "XRP" "DOGE" "TON11419" "ADA" "SHIB" "AVAX" "WSTETH" "WETH" "DOT" "LINK" "WBTC" "TRX" "WTRX")
exp_name="cryptopulse_results"
log_folder="./logs/${exp_name}"
if [ ! -d "$log_folder" ]; then
    mkdir -p "$log_folder"
fi


for data in "${top20[@]}"
do
    data="${data}-USD"
    filename="${log_folder}/${data}.log"
    echo "Working on ${data} now ... See the results in 'results/${exp_name}'"
    python -u main.py \
        --data  "$data" \
        --use-cuda \
        --exp-name "$exp_name" \
        --ob-len $ob_len \
        --pred-len $pred_len \
        --batch-size $batch_size \
        --learning-rate 0.0005 > "$filename" 
done

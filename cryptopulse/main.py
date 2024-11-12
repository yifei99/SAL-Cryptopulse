import shutil
import argparse
import textwrap
import pandas as pd
from cryptopulse.workflow import Workflow

# Argument parser setup
parser = argparse.ArgumentParser()

def _add_hanging_indent(text, indent_size=4):
    terminal_width = shutil.get_terminal_size().columns
    text_width = terminal_width - indent_size
    indent = " " * indent_size
    wrapper = textwrap.TextWrapper(
        width=text_width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=True,
        replace_whitespace=False,
    )
    wrapped_text = wrapper.fill(text)
    return wrapped_text

def print_settings(content):
    terminal_width = shutil.get_terminal_size().columns
    print("Settings:" + ">" * (terminal_width - 9))
    print(_add_hanging_indent(content))
    print("End of Settings:" + "<" * (terminal_width - 16))

# Argument definitions
parser.add_argument("--ob-len", type=int, default=96, help="Length of observation sequence.")
parser.add_argument("--pred-len", type=int, default=96, help="Length of prediction sequence.")
parser.add_argument("--data", type=str, required=True, help="Dataset to use")
parser.add_argument("--enc-in-channels", type=int, default=-1, help="Channels of input for encoder.")
parser.add_argument("--dec-in-channels", type=int, default=-1, help="Channels of input for decoder.")
parser.add_argument("--dec-out-channels", type=int, default=-1, help="Channels of the output.")
parser.add_argument("--d-model", type=int, default=512, help="Dimension of model.")
parser.add_argument("--n-heads", type=int, default=8, help="Num of heads")
parser.add_argument("--train-epochs", type=int, default=10, help="Training epochs.")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
parser.add_argument("--loss", type=str, default="mse", help="Loss function")
parser.add_argument(
    "--lr-adj",
    type=str,
    default="halving_decay",
    help=(
        "Strategy to adjust learning rate. "
        "halving_decay: decay factor is 0.5 * (epoch - 1). "
        "epoch_hard_threshold_x: 0.1 * lr once epoch > x. "
        "fixed: {epoch: a specific lr}."
    ),
)
parser.add_argument("--exp-name", type=str, default="Test", help="Name of this experiment.")
parser.add_argument("--use-cuda", action="store_true", help="Use CUDA.")

def main():
    args = parser.parse_args()
    print_settings(str(args))

    terminal_width = shutil.get_terminal_size().columns
    res_df, result_summary_path = [], None

    for exp_i in range(1, 6):
        workflow = Workflow(args)  # set up experiments
        result_summary_path = workflow.result_folder

        print("Training" + ">" * (terminal_width - 8))
        workflow.train(exp_i)
        print("End of Training" + "<" * (terminal_width - 15))

        print("Testing" + ">" * (terminal_width - 7))
        res_df_step = workflow.test(exp_i)
        res_df.append(res_df_step)
        print("End of Testing" + "<" * (terminal_width - 14))

    pd.concat(res_df).to_csv(result_summary_path / "result_summary.csv", index=False)

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()

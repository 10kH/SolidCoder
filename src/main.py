import dotenv
dotenv.load_dotenv()

import argparse
import sys
from datetime import datetime
from constants.paths import *

from models.Gemini import Gemini
from models.OpenAI import OpenAIModel

from results.Results import Results

from promptings.PromptingFactory import PromptingFactory
from datasets.DatasetFactory import DatasetFactory
from models.ModelFactory import ModelFactory

from constants.verboseType import *

from utils.summary import gen_summary
from utils.runEP import run_eval_plus
from utils.evaluateET import generate_et_dataset_human
from utils.evaluateET import generate_et_dataset_mbpp
from utils.generateEP import generate_ep_dataset_human
from utils.generateEP import generate_ep_dataset_mbpp

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="HumanEval",
    choices=[
        "HumanEval",
        "MBPP",
        "APPS",
        "xCodeEval",
        "CC",
    ]
)
parser.add_argument(
    "--strategy",
    type=str,
    default="Direct",
    choices=[
        "Direct",
        "CoT",
        "SelfPlanning",
        "Analogical",
        "MapCoder",
        "CodeSIM",
        "SolidCoder",
        "CodeSIMWD",
        "CodeSIMWPV",
        "CodeSIMWPVD",
        "CodeSIMA",
        "CodeSIMC",
    ]
)
parser.add_argument(
    "--model",
    type=str,
    default="ChatGPT",
)
parser.add_argument(
    "--model_provider",
    type=str,
    default="OpenAI",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.95
)
parser.add_argument(
    "--pass_at_k",
    type=int,
    default=1
)
parser.add_argument(
    "--language",
    type=str,
    default="Python3",
    choices=[
        "C",
        "C#",
        "C++",
        "Go",
        "PHP",
        "Python3",
        "Ruby",
        "Rust",
    ]
)

parser.add_argument(
    "--cont",
    type=str,
    default="yes",
    choices=[
        "yes",
        "no"
    ]
)

parser.add_argument(
    "--result_log",
    type=str,
    default="partial",
    choices=[
        "full",
        "partial"
    ]
)

parser.add_argument(
    "--verbose",
    type=str,
    default="2",
    choices=[
        "2",
        "1",
        "0",
    ]
)

parser.add_argument(
    "--store_log_in_file",
    type=str,
    default="yes",
    choices=[
        "yes",
        "no",
    ]
)

parser.add_argument(
    "--max_assumption_rounds",
    type=int,
    default=3,
    help="Max assumption breaking rounds (SolidCoder, default: 3)"
)

# SolidCoder-specific arguments
parser.add_argument("--enable_shift_left", action="store_true", help="Enable Shift-Left Planning (SolidCoder)")
parser.add_argument("--enable_oracle_assert", action="store_true", help="Enable Oracle-based Assertions (SolidCoder)")
parser.add_argument("--enable_live_verify", action="store_true", help="Enable Live Concrete Verification (SolidCoder)")
parser.add_argument("--enable_inter_sim", action="store_true", help="Enable Intermediate Code Simulation (SolidCoder)")
parser.add_argument("--enable_defensive_test", action="store_true", help="Enable Defensive Accumulation (SolidCoder)")

args = parser.parse_args()

DATASET = args.dataset
STRATEGY = args.strategy
MODEL_NAME = args.model
MODEL_PROVIDER_NAME = args.model_provider
TEMPERATURE = args.temperature
TOP_P = args.top_p
PASS_AT_K = args.pass_at_k
LANGUAGE = args.language
CONTINUE = args.cont
RESULT_LOG_MODE = args.result_log
VERBOSE = int(args.verbose)
STORE_LOG_IN_FILE = args.store_log_in_file
MAX_ASSUMPTION_ROUNDS = args.max_assumption_rounds
ENABLE_SHIFT_LEFT = args.enable_shift_left
ENABLE_ORACLE_ASSERT = args.enable_oracle_assert
ENABLE_LIVE_VERIFY = args.enable_live_verify
ENABLE_INTER_SIM = args.enable_inter_sim
ENABLE_DEFENSIVE_TEST = args.enable_defensive_test

MODEL_NAME_FOR_RUN = MODEL_NAME

# SolidCoder: Auto-generate suffix from disabled flags for ablation experiments
if STRATEGY == "SolidCoder":
    disabled_flags = []
    if not ENABLE_SHIFT_LEFT:
        disabled_flags.append("woS")
    if not ENABLE_ORACLE_ASSERT:
        disabled_flags.append("woO")
    if not ENABLE_LIVE_VERIFY:
        disabled_flags.append("woL")
    if not ENABLE_INTER_SIM:
        disabled_flags.append("woI")
    if not ENABLE_DEFENSIVE_TEST:
        disabled_flags.append("woD")

    if disabled_flags:
        MODEL_NAME_FOR_RUN = f"{MODEL_NAME}-{'-'.join(disabled_flags)}"

RUN_NAME = f"results/{DATASET}/{STRATEGY}/{MODEL_NAME_FOR_RUN}/{LANGUAGE}-{TEMPERATURE}-{TOP_P}-{PASS_AT_K}"

run_no = 1
while os.path.exists(f"{RUN_NAME}/Run-{run_no}"):
    run_no += 1

if CONTINUE == "yes" and run_no > 1:
    run_no -= 1

RUN_NAME = f"{RUN_NAME}/Run-{run_no}"

if not os.path.exists(RUN_NAME):
    os.makedirs(RUN_NAME)

RESULTS_PATH = f"{RUN_NAME}/Results.jsonl"
SUMMARY_PATH = f"{RUN_NAME}/Summary.txt"
LOGS_PATH = f"{RUN_NAME}/Log.txt"

if STORE_LOG_IN_FILE.lower() == 'yes':
    sys.stdout = open(
        LOGS_PATH,
        mode="a",
        encoding="utf-8",
        buffering=1  # Line buffering - flush after every newline
    )

if CONTINUE == "no" and VERBOSE >= VERBOSE_MINIMAL:
    print(f"""
##################################################
Experiment start {RUN_NAME}, Time: {datetime.now()}
###################################################
""")

extra_kwargs = {}
if STRATEGY == "SolidCoder":
    extra_kwargs["max_assumption_rounds"] = MAX_ASSUMPTION_ROUNDS
    extra_kwargs["enable_shift_left"] = ENABLE_SHIFT_LEFT
    extra_kwargs["enable_oracle_assert"] = ENABLE_ORACLE_ASSERT
    extra_kwargs["enable_live_verify"] = ENABLE_LIVE_VERIFY
    extra_kwargs["enable_inter_sim"] = ENABLE_INTER_SIM
    extra_kwargs["enable_defensive_test"] = ENABLE_DEFENSIVE_TEST

strategy = PromptingFactory.get_prompting_class(STRATEGY)(
    model=ModelFactory.get_model_class(MODEL_PROVIDER_NAME)(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        top_p=TOP_P
    ),
    data=DatasetFactory.get_dataset_class(DATASET)(),
    language=LANGUAGE,
    pass_at_k=PASS_AT_K,
    results=Results(RESULTS_PATH),
    verbose=VERBOSE,
    **extra_kwargs
)

strategy.run(RESULT_LOG_MODE.lower() == 'full')

if VERBOSE >= VERBOSE_MINIMAL:
    print(f"""
##################################################
Experiment end {RUN_NAME}, Time: {datetime.now()}
###################################################
""")

gen_summary(RESULTS_PATH, SUMMARY_PATH)

ET_RESULTS_PATH = f"{RUN_NAME}/Results-ET.jsonl"
ET_SUMMARY_PATH = f"{RUN_NAME}/Summary-ET.txt"

EP_RESULTS_PATH = f"{RUN_NAME}/Results-EP.jsonl"
EP_SUMMARY_PATH = f"{RUN_NAME}/Summary-EP.txt"

if "human" in DATASET.lower():
    generate_et_dataset_human(RESULTS_PATH, ET_RESULTS_PATH)
    gen_summary(ET_RESULTS_PATH, ET_SUMMARY_PATH)

    # generate_ep_dataset_human(RESULTS_PATH, EP_RESULTS_PATH)
    # run_eval_plus(EP_RESULTS_PATH, EP_SUMMARY_PATH, "humaneval")

elif "mbpp" in DATASET.lower():
    generate_et_dataset_mbpp(RESULTS_PATH, ET_RESULTS_PATH)
    gen_summary(ET_RESULTS_PATH, ET_SUMMARY_PATH)

    # generate_ep_dataset_human(RESULTS_PATH, EP_RESULTS_PATH)
    # run_eval_plus(EP_RESULTS_PATH, EP_SUMMARY_PATH, "mbpp")

if STORE_LOG_IN_FILE.lower() == 'yes':
    sys.stdout.close()

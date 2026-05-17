from pathlib import Path
from task import Task
import argparse
from .experiments import EXPERIMENTS, get_dfs_for_all_models
from model import MODELS
from .fragile import (
    get_absolute_fragile,
    get_relative_drop_fragile,
    get_rmce_fragile,
    get_strongly_fragile,
    get_cross_model_fragile,
    get_cross_model_df,
)
from .methods import calculate_relative_drop
from .clustering import run_clustering
from .definitions import DEFINITIONS, FragileDefinition


TASK_NAME = "fragile"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Analyze fragile classes")
    parser.add_argument(
        "--exp",
        type=str,
        default="all_corruptions",
        help="Experiment",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="results",
        help="path to data",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--definition",
        type=str,
        default="ab",
        choices=list(DEFINITIONS.keys()) + ["all"],
        help="Fragile class combination definition (use 'all' to run each definition in sequence)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run all experiments × definitions and save results to disk",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Base output directory for sweep results",
    )


def run(args: argparse.Namespace):
    if args.sweep:
        run_sweep(args)
        return

    variations = EXPERIMENTS[args.exp]
    dfs = get_dfs_for_all_models(variations, args.data_path)

    definitions = list(DEFINITIONS.values()) if args.definition == "all" else [DEFINITIONS[args.definition]]

    if args.model:
        for definition in definitions:
            print(f"\n=== {definition.label} ===")
            df = _get_fragile(dfs[args.model], dfs["alexnet"], definition)
            print(df[df["is_strongly_fragile"] == 1])
            print(df.describe())
        return

    for definition in definitions:
        print(f"\n=== {definition.label} ===")
        fragile_dfs = [_get_fragile(df, dfs["alexnet"], definition) for df in dfs.values()]
        cross = get_cross_model_fragile(fragile_dfs, definition, min_models=15)
        print(cross)
        print(len(cross))

    print("HDBSCAN")
    across_models_df = get_cross_model_df(fragile_dfs)
    print(across_models_df.sort_values("fragile_count", ascending=False))
    print(len(across_models_df))

    run_clustering(across_models_df)


def run_sweep(args: argparse.Namespace):
    out = Path(args.output_dir)

    for exp_name, variations in EXPERIMENTS.items():
        print(f"\n[sweep] experiment: {exp_name}")
        dfs = get_dfs_for_all_models(variations, args.data_path)

        for def_name, definition in DEFINITIONS.items():
            print(f"  definition: {definition.label}")

            if args.model:
                df = _get_fragile(dfs[args.model], dfs["alexnet"], definition)
                dest = out / "fragile" / exp_name / def_name / args.model
                dest.mkdir(parents=True, exist_ok=True)
                df[df["is_strongly_fragile"] == 1].to_csv(dest / "fragile_classes.csv", index=False)
            else:
                fragile_dfs = [_get_fragile(df, dfs["alexnet"], definition) for df in dfs.values()]
                cross = get_cross_model_fragile(fragile_dfs, definition, min_models=15)
                dest = out / "fragile" / exp_name / def_name
                dest.mkdir(parents=True, exist_ok=True)
                cross.to_csv(dest / "cross_model.csv", index=False)


def _get_fragile(df, alexnet_df, definition: FragileDefinition):
    df = calculate_relative_drop(df)
    df_a = get_absolute_fragile(df)
    df_b = get_relative_drop_fragile(df)
    df_c = get_rmce_fragile(df, alexnet_df)
    strong_fragile = get_strongly_fragile(df_a, df_b, df_c, definition)

    print(df.head())
    super_giga_fragile = strong_fragile[strong_fragile["is_strongly_fragile"] == 1]
    print(super_giga_fragile)
    print(len(super_giga_fragile))

    return df.merge(strong_fragile, on="synset")



# def _get_fragile(model, variations, data_path):
#     df, alexnet_df = build_df_per_class(model, variations, data_path)

#     df_a = get_absolute_fragile(df)
#     df_b = get_relative_drop_fragile(df)
#     df_c = get_rmce_fragile(df, alexnet_df)

#     strong_fragile = get_strongly_fragile(df_a, df_b, df_c)
#     df = df.merge(strong_fragile, on="synset")

#     print(df.head())

#     super_giga_fragile = df[df["is_strongly_fragile"] == 1]
#     print(super_giga_fragile)
#     print(len(super_giga_fragile))

#     return df

# print("---")

# df_a_b = df[(df["is_fragile_a"] == 1) & (df["is_fragile_b"] == 1)][["synset", "acc_clean", "acc_corrupt", "rel_drop", "RmCE"]]

# print("a i b")
# print(df_a_b.head(5))
# print(df_a_b.describe())
# print(len(df_a_b))

# df_a_c = df[(df["is_fragile_a"] == 1) & (df["is_fragile_c"] == 1)][["synset", "acc_clean", "acc_corrupt", "rel_drop", "RmCE"]]

# print("a i c")
# print(df_a_c.head(5))
# print(df_a_c.describe())
# print(len(df_a_c))

# df_b_c = df[(df["is_fragile_b"] == 1) & (df["is_fragile_c"] == 1)][["synset", "acc_clean", "acc_corrupt", "rel_drop", "RmCE"]]
# print("b i c")
# print(df_b_c.head(5))
# print(df_b_c.describe())
# print(len(df_b_c))


# klasa która spada nieproporcjonalnie (B) ORAZ albo była dobra i się posypała (A) albo jest gorsza niż AlexNet (C) B ∩ (A ∪ C).

#         synset  y_true  acc_clean  acc_corrupt  rel_drop      RmCE  fragile_count
# 16   n01737021      58   0.834737     0.323228  0.618620  2.264649             19
# 127  n02971356     478   0.577895     0.246063  0.576744  4.957643             19
# 34   n01985128     124   0.768421     0.280547  0.645239  4.345668             19
# 149  n03141823     523   0.675789     0.314512  0.541183  2.393621             19
# 263  n04154565     784   0.543158     0.227411  0.584867  5.456464             19
# 213  n03793489     673   0.470526     0.208028  0.564571  2.163447             19
# 183  n03532672     600   0.455789     0.189811  0.593777  4.841850             19
# 361  n15075141     999   0.595789     0.183032  0.700778  3.165321             19
# 273  n04254120     804   0.718947     0.368632  0.496115  3.980861             19
# 270  n04208210     792   0.753684     0.338400  0.560751  2.331311             19
# 284  n04332243     828   0.693684     0.336435  0.519060  1.978854             19
# 349  n07860988     961   0.515789     0.140168  0.731853  2.537980             19
# 320  n04557648     898   0.629474     0.283214  0.552617  3.763692             19
# 316  n04548362     893   0.695789     0.298414  0.579739  2.098814             19
# 305  n04493381     876   0.402105     0.182049  0.546747  2.378128             19
# 238  n03995372     740   0.683158     0.294751  0.575201  2.954414             19
# 266  n04192698     787   0.700000     0.353179  0.500802  2.864711             19
# 275  n04263257     809   0.661053     0.079509  0.878785  2.262230             19
# 135  n03041632     499   0.642222     0.290311  0.558382  3.097809             18
# 141  n03089624     509   0.701111     0.317541  0.547778  2.505904             18
# 19   n01755581      67   0.838889     0.268533  0.684423  1.682796             18
# 14   n01698640      50   0.917778     0.377719  0.591222  1.129516             18
# 5    n01641577      30   0.824444     0.390237  0.526236  0.793895             18
# 255  n04116512     767   0.695556     0.340667  0.515748  1.914868             18
# 352  n07880968     965   0.905556     0.374089  0.590530  1.227217             18
# 288  n04357314     838   0.553333     0.271896  0.523382  2.605898             18
# 243  n04026417     748   0.608889     0.250341  0.593431  2.182720             18
# 227  n03908714     710   0.722222     0.336415  0.543577  2.075721             18
# 201  n03759954     650   0.655294     0.303827  0.546140  1.984940             17
# 217  n03832673     681   0.438824     0.126447  0.713386  2.632386             17
# 262  n04141975     778   0.801176     0.346180  0.579718  1.679365             17
# 324  n04597913     910   0.817647     0.325537  0.607460  1.992885             17
# 293  n04376876     845   0.721176     0.340063  0.537183  1.923521             17
# 231  n03938244     721   0.907059     0.330965  0.637074  1.602636             17
# 100  n02769748     414   0.494118     0.144486  0.709132  2.029594             17
# 175  n03481172     587   0.702353     0.302839  0.579371  1.963534             17
# 122  n02910353     464   0.603529     0.297271  0.513938  3.555636             17
# 188  n03633091     618   0.560000     0.195671  0.653495  2.754507             17
# 18   n01742172      61   0.846250     0.418800  0.507195  0.809973             16
# 13   n01697457      49   0.885000     0.377850  0.573879  1.252841             16
# 0    n01498041       6   0.906250     0.318917  0.647814  1.302484             16
# 192  n03666591     626   0.831250     0.394917  0.533737  1.724183             16
# 216  n03814906     679   0.940000     0.419317  0.554828  0.850049             16
# 267  n04200800     788   0.907500     0.385600  0.577278  1.246576             16
# 281  n04325704     824   0.703750     0.378050  0.461607  2.775852             16
# 98   n02730930     411   0.823750     0.359700  0.566429  1.510579             16
# 161  n03291819     549   0.625000     0.269117  0.574052  1.772327             16
# 180  n03498962     596   0.728000     0.377191  0.494722  2.332506             15
# 165  n03372029     558   0.569333     0.305671  0.467936  2.777341             15
# 159  n03255030     543   0.797333     0.421031  0.481407  2.191201             15
# 131  n03000134     489   0.828000     0.381049  0.540537  1.654557             15
# 113  n02840245     446   0.745333     0.374027  0.506057  1.936579             15
# 104  n02791270     424   0.736000     0.223929  0.700470  1.798002             15
# 304  n04479046     869   0.726667     0.358222  0.509030  1.722776             15
# 332  n07614500     928   0.724000     0.159129  0.782048  1.763752             15
# 55


# A and B not C
#         synset  y_true  acc_clean  acc_corrupt  rel_drop      RmCE  fragile_count
# 5    n01641577      30   0.824444     0.390237  0.526236  0.793895             18
# 13   n01698640      50   0.917778     0.377719  0.591222  1.129516             18
# 0    n01498041       6   0.906250     0.318917  0.647814  1.302484             16
# 12   n01697457      49   0.885000     0.377850  0.573879  1.252841             16
# 15   n01742172      61   0.846250     0.418800  0.507195  0.809973             16
# 151  n03814906     679   0.940000     0.419317  0.554828  0.850049             16
# 181  n04200800     788   0.907500     0.385600  0.577278  1.246576             16
# 232  n07880968     965   0.914667     0.412409  0.551002  1.159770             15
# 8


# A and B
#         synset  y_true  acc_clean  acc_corrupt  rel_drop      RmCE  fragile_count
# 5    n01641577      30   0.824444     0.390237  0.526236  0.793895             18
# 13   n01698640      50   0.917778     0.377719  0.591222  1.129516             18
# 275  n07880968     965   0.905556     0.374089  0.590530  1.227217             18
# 185  n03938244     721   0.907059     0.330965  0.637074  1.602636             17
# 0    n01498041       6   0.906250     0.318917  0.647814  1.302484             16
# 12   n01697457      49   0.885000     0.377850  0.573879  1.252841             16
# 174  n03814906     679   0.940000     0.419317  0.554828  0.850049             16
# 16   n01742172      61   0.846250     0.418800  0.507195  0.809973             16
# 211  n04200800     788   0.907500     0.385600  0.577278  1.246576             16
# 253  n04597913     910   0.848000     0.347360  0.591815  2.027430             15
# 10

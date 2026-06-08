import subprocess

commands = [
    [
        "./run.sh", "src/main.py", "adversarial",
        "--model", "vit_b_16",
        "--batch-size", "16",
        "--num-workers", "8",
        "--output-name", "vit_b"
    ],
    # [
    #     "./run.sh", "src/main.py", "adversarial",
    #     "--model", "maxvit_t",
    #     "--fragile", "n02097658,n04254120,n03223299,n10148035,n04116512",
    #     "--robust", "n04525305,n02268853,n03874293,n01806143,n04310018",
    #     "--batch-size", "16",
    #     "--num-workers", "8",
    #     "--output-name", "weather"
    # ],
    # [
    #     "./run.sh", "src/main.py", "adversarial",
    #     "--model", "maxvit_t",
    #     "--fragile", "n04023962,n04141975,n03873416,n04204238,n01943899",
    #     "--robust", "n01582220,n04019541,n04146614,n01833805,n02787622",
    #     "--batch-size", "16",
    #     "--num-workers", "8",
    #     "--output-name", "noise"
    # ],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    print(f"Finished: {' '.join(cmd)} (exit code: {result.returncode})\n")
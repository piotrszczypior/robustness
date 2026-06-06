import subprocess

commands = [
    [
        "./run.sh", "src/main.py", "adversarial",
        "--model", "maxvit_t",
        "--fragile", "n03763968,n03938244,n03220513,n04127249,n04507155",
        "--robust", "n01629819,n02088094,n02326432,n02086910,n01530575",
        "--batch-size", "16",
        "--num-workers", "8",
        "--output-name", "digital"
    ],
    [
        "./run.sh", "src/main.py", "adversarial",
        "--model", "maxvit_t",
        "--fragile", "n01770081,n02106550,n01770393,n01695060,n03425413",
        "--robust", "n04273569,n02480495,n02817516,n02437312,n12768682",
        "--batch-size", "16",
        "--num-workers", "8",
        "--output-name", "blur"
    ],
    [
        "./run.sh", "src/main.py", "adversarial",
        "--model", "maxvit_t",
        "--fragile", "n02097658,n04254120,n03223299,n10148035,n04116512",
        "--robust", "n04525305,n02268853,n03874293,n01806143,n04310018",
        "--batch-size", "16",
        "--num-workers", "8",
        "--output-name", "weather"
    ],
    [
        "./run.sh", "src/main.py", "adversarial",
        "--model", "maxvit_t",
        "--fragile", "n04023962,n04141975,n03873416,n04204238,n01943899",
        "--robust", "n01582220,n04019541,n04146614,n01833805,n02787622",
        "--batch-size", "16",
        "--num-workers", "8",
        "--output-name", "noise"
    ],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    print(f"Finished: {' '.join(cmd)} (exit code: {result.returncode})\n")
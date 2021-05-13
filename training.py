import sys
from main import main as do
from tabulate import tabulate

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} cfg-name")
    sys.exit()

runs = 5
rows = []
headers = None
name = sys.argv[1]

print(f"Training {runs} of {name}")

for i in range(1, runs + 1):
    rs, headers = do(f'cfg/{name}.json', seed=10 + i, output_dir=f'neurips-mixed-rv/{name}/{i}')

    with open(f'neurips-mixed-rv/{name}/{i}/validation.txt', 'w') as f:
        print(tabulate(rs, headers=headers), file=f)

    rows.extend(rs)

print(tabulate(rows, headers=headers))

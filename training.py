import sys
from main import main as do
from tabulate import tabulate

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} cfg-name first last")
    sys.exit()

name = sys.argv[1]
first = int(sys.argv[2])
last = int(sys.argv[3])
rows = []
headers = None

print(f"Training {name} {first}-{last}")

for i in range(first, last + 1):
    rs, headers = do(f'cfg/{name}.json', seed=10 + i, output_dir=f'neurips-mixed-rv/{name}/{i}')

    with open(f'neurips-mixed-rv/{name}/{i}/validation.txt', 'w') as f:
        print(tabulate(rs, headers=headers), file=f)

    rows.extend(rs)

print(tabulate(rows, headers=headers))

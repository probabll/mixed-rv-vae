import main
import argparse
import json

known = main.default_cfg()

parser = argparse.ArgumentParser()
parser.add_argument('json', type=str)
args, unknownargs = parser.parse_known_args()
kw = dict((k.replace('--', ''), unknownargs[i+1]) for i, k in enumerate(unknownargs) if i%2 == 0)
typed_kw = dict()
for k, v in kw.items():
    if k not in known:
        raise ValueError(f"Unknown option {k}")
    else:
        t = type(known[k])
        typed_kw[k] = t(v)
main.main(main.load_cfg(args.json, **typed_kw))

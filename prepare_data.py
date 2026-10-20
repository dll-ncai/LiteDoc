"""
Convert the raw document-understanding benchmarks into LiteDoc JSONL files.

Examples:
    python prepare_data.py docvqa  --annotation_file /data/docvqa/val_v1.0.json --images_dir /data/docvqa \
                                   --output data/docvqa_val.jsonl
    python prepare_data.py infovqa --annotation_file /data/infovqa/infographicsVQA_val_v1.0.json \
                                   --images_dir /data/infovqa/images --output data/infovqa_val.jsonl
    python prepare_data.py cord    --source naver-clova-ix/cord-v2 --split test --output data/cord_test.jsonl
    python prepare_data.py sroie   --root /data/sroie/test --output data/sroie_test.jsonl
    python prepare_data.py funsd   --root /data/funsd/dataset/testing_data --output data/funsd_test.jsonl
    python prepare_data.py docile  --root /data/docile --split val --output data/docile_val.jsonl
    python prepare_data.py klc     --root /data/kleister-charity --split dev-0 --output data/klc_dev.jsonl
    python prepare_data.py wtq     --root /data/WikiTableQuestions --output data/wtq_test.jsonl
    python prepare_data.py rvl_cdip --root /data/rvl-cdip --output data/rvl_cdip_test.jsonl
"""

import argparse
import os
import random

from litedoc import data


def main():
    parser = argparse.ArgumentParser(description="Convert DU benchmarks to LiteDoc JSONL")
    sub = parser.add_subparsers(dest="dataset", required=True)

    def add(name, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--output", required=True, help="output .jsonl path")
        p.add_argument("--work_dir", default=None,
                       help="where rendered/extracted images are cached (default: next to --output)")
        p.add_argument("--max_samples", type=int, default=-1, help="keep at most N samples (-1 = all)")
        p.add_argument("--shuffle", action="store_true", help="shuffle before truncating to --max_samples")
        p.add_argument("--seed", type=int, default=42)
        return p

    p = add("docvqa", "DocVQA (SP-DocVQA) official json")
    p.add_argument("--annotation_file", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--suffix", default=data.VQA_SUFFIX)

    p = add("infovqa", "InfographicsVQA official json")
    p.add_argument("--annotation_file", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--suffix", default=data.VQA_SUFFIX)

    p = add("cord", "CORD v2 (HuggingFace dataset id or local path)")
    p.add_argument("--source", default="naver-clova-ix/cord-v2")
    p.add_argument("--split", default="test")

    p = add("sroie", "SROIE task-3 folder with img/ and entities/")
    p.add_argument("--root", required=True)

    p = add("funsd", "FUNSD testing_data folder with images/ and annotations/")
    p.add_argument("--root", required=True)

    p = add("docile", "DocILE dataset root")
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--max_pages", type=int, default=1)

    p = add("klc", "Kleister Charity repository root (documents fetched with git-annex)")
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="dev-0")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--max_pages", type=int, default=None,
                   help="cap the number of rendered pages per report (default: all pages)")

    p = add("wtq", "WikiTableQuestions repository root")
    p.add_argument("--root", required=True)
    p.add_argument("--split_file", default="data/pristine-unseen-tables.tsv")
    p.add_argument("--renderer", choices=["auto", "selenium", "pil"], default="auto")
    p.add_argument("--suffix", default=data.VQA_SUFFIX)

    p = add("rvl_cdip", "RVL-CDIP root with labels/ and images/")
    p.add_argument("--root", required=True)
    p.add_argument("--labels_file", default="labels/test.txt")
    p.add_argument("--images_subdir", default="images")

    args = parser.parse_args()
    work_dir = args.work_dir or os.path.dirname(os.path.abspath(args.output))

    if args.dataset in ("docvqa", "infovqa"):
        samples = data.convert_docvqa(args.annotation_file, args.images_dir, dataset=args.dataset, suffix=args.suffix)
    elif args.dataset == "cord":
        samples = data.convert_cord(args.source, work_dir, split=args.split)
    elif args.dataset == "sroie":
        samples = data.convert_sroie(args.root)
    elif args.dataset == "funsd":
        samples = data.convert_funsd(args.root)
    elif args.dataset == "docile":
        samples = data.convert_docile(args.root, work_dir, split=args.split, dpi=args.dpi, max_pages=args.max_pages)
    elif args.dataset == "klc":
        samples = data.convert_klc(args.root, work_dir, split=args.split, dpi=args.dpi, max_pages=args.max_pages)
    elif args.dataset == "wtq":
        samples = data.convert_wtq(args.root, work_dir, split_file=args.split_file, renderer=args.renderer,
                                   suffix=args.suffix)
    elif args.dataset == "rvl_cdip":
        samples = data.convert_rvl_cdip(args.root, labels_file=args.labels_file, images_subdir=args.images_subdir)
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(args.dataset)

    if args.max_samples > 0 and not args.shuffle:
        samples = (s for i, s in zip(range(args.max_samples), samples))
    samples = list(samples)
    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
        if args.max_samples > 0:
            samples = samples[:args.max_samples]

    count = data.write_jsonl(samples, args.output)
    print(f"wrote {count} {args.dataset} samples to {args.output}")


if __name__ == "__main__":
    main()

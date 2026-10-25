"""Command-line entry points for preparing, fitting, and querying NORMST."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="normst", description="Histology-guided spatial transcriptomics super-resolution")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-visium", help="prepare classic Visium Space Ranger counts and H&E")
    prepare.add_argument("--matrix", required=True, help="filtered_feature_bc_matrix.h5")
    prepare.add_argument("--positions", required=True, help="tissue_positions.csv or tissue_positions_list.csv")
    prepare.add_argument("--image", required=True, help="RGB H&E image in full-resolution Space Ranger coordinates")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--genes", type=int, default=2000)
    prepare.add_argument("--gene-list", help="optional text file with one gene ID per line, in output order")

    features = commands.add_parser("extract", help="extract frozen Virchow2 image features")
    features.add_argument("--data", required=True, help="dataset.json")
    features.add_argument("--weights", required=True, help="local Virchow2 model.safetensors")
    features.add_argument("--device", default="cuda:0")
    features.add_argument("--batch-tiles", type=int, default=8)

    train = commands.add_parser("train", help="fit the final expression model to all supplied spots")
    train.add_argument("--data", required=True, help="dataset.json")
    train.add_argument("--output", required=True)
    train.add_argument("--config", help="optional JSON overrides of the default training settings")
    train.add_argument("--device", default="cuda:0")

    predict = commands.add_parser("predict", help="export count estimates on a grid or in query circles")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--bin-um", type=int, default=8)
    predict.add_argument("--mask", help="optional boolean NPY mask on the requested output grid")
    predict.add_argument("--queries", help="optional NPZ with centers_yx_um and radius_um instead of a grid")
    predict.add_argument("--batch", type=int, default=128)

    args = parser.parse_args(argv)
    if args.command == "prepare-visium":
        from .prepare import prepare_visium
        path = prepare_visium(args.matrix, args.positions, args.image, args.output,
                              n_genes=args.genes, gene_list=args.gene_list)
    elif args.command == "extract":
        from .features import extract
        path = extract(args.data, args.weights, device=args.device, batch_tiles=args.batch_tiles)
    elif args.command == "train":
        from .train import fit
        path = fit(args.data, args.output, device=args.device, config=args.config)
    else:
        from .predict import predict as export
        path = export(args.checkpoint, args.output, device=args.device, bin_um=args.bin_um,
                      mask=args.mask, queries=args.queries, batch=args.batch)
    print(path)

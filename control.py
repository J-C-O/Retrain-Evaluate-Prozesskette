import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect", help="detect a face",
                        action="store_true")
    parser.add_argument("--censor", help="detect a face and censor it",
                        action="store_true")
    parser.add_argument("--retrain", help="retrain the model",
                        action="store_true")

    args = parser.parse_args()
    if args.detect:
        from scripts.detect import main
        main()
    if args.censor:
        from scripts.censor import main
        main()
    if args.retrain:
        from scripts.retrain import main
        main()

if __name__ == '__main__':
    main()

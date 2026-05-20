from utils.eval_utils import parse_args

def main():
    args = parse_args()
    print(args)
    print(args.sampling_temp)

if __name__ == "__main__":
    main()

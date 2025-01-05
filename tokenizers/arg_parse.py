import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The file to read from")
    parser.add_argument("-n", type=int, default=10, help="Number of characters to print")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        print(f.read(args.n))

if __name__ == "__main__":
    main()

# usage: python arg_parse.py -n 500 the-verdict.txt

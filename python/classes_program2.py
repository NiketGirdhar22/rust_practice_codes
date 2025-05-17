import sys

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <infile> <outfile>")
    sys.exit(1)

infile_name = sys.argv[1]
outfile_name = sys.argv[2]

try:
    with open(infile_name, 'r') as infile:
        try:
            with open(outfile_name, 'w') as outfile:
                for line in infile:
                    outfile.write(line)
        except IOError:
            print(f"Oops! Could not open {outfile_name} for writing")
            sys.exit(1)
except IOError:
    print(f"Oops! Could not read {infile_name}")
    sys.exit(1)

print(f"Copied contents from {infile_name} to {outfile_name}")
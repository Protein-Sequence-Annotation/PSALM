from parsers import train_parser

parser = train_parser()

a = parser.parse_args()

print(a.num_shards)
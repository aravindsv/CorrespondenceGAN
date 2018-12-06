import os
import shutil
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('datadir')
parser.add_argument('output_dir')
args = parser.parse_args()

filelist = defaultdict(list)

os.makedirs(args.output_dir, exist_ok=True)
for idx, file in enumerate(os.listdir(args.datadir)):
	obj = file.split('_')[0]
	filelist[obj].append(os.path.abspath(os.path.join(args.datadir, file)))

for obj in filelist.keys():
	print("Processing object {}".format(obj))
	if os.path.isdir(os.path.join(args.output_dir, obj)):
		shutil.rmtree(os.path.join(args.output_dir))
	os.makedirs(os.path.join(args.output_dir, obj))
	final_dst = os.path.join(os.path.join(args.output_dir, obj), 'imgs')
	os.makedirs(final_dst)
	for file in filelist[obj]:
		shutil.copy(file, final_dst)

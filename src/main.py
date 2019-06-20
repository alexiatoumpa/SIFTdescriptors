from __future__ import print_function
import __init__
import sys
sys.path.append('/home/scat/OneDrive/Github/SIFTdescriptors/func/')

from parse_images import ProcessImages

import time

def main():

	start = time.time()
	__init__.init()
	ProcessImages()
	end = time.time() - start # time in seconds
	print("TIME OF EXECUTION: ", end/60)


if __name__== "__main__":
	# Call main
	main()


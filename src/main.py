#
# Copyright (C) 2018 University of Leeds
# Author: Alexia Toumpa
# email: scat@leeds.ac.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>
#

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


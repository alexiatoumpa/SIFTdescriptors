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
sys.path.append('/home/scat/OneDrive/Github/object_affordances/func/')
from parse_images import ProcessImages

import time




def main(unique_graph_name, detect3D, filtering_window, show_qsr, show_image):

	start = time.time()
	__init__.init()

	ProcessImages(unique_graph_name, detect3D, filtering_window, show_qsr, show_image)
	end = time.time() - start # time in seconds
	print("TIME OF EXECUTION: ", end/60)


if __name__== "__main__":

	# Input parameters
	unique_graph_name = sys.argv[1]
	filtering_window = int(sys. argv[3])
	if sys.argv[2] == 'T':
		detect3D = True
		print("Compute WITH ED RCC.\n")
	elif sys.argv[2] == 'F':
		detect3D = False
		print("Compute WITHOUT ED RCC.\n")
	else:
		print("The second input is wrong. Please type the letter T or F, for True or False accordingly.")
		exit()

	if sys.argv[4] == 'T':
		show_qsr = True
		print("Show QSRs.\n")
	elif sys.argv[4] == 'F':
		show_qsr = False
		print("Do not show QSRs.\n")
	else:
		print("The forth input is wrong. Please type the letter T or F, for True or False accordingly.")
		exit()

	if sys.argv[5] == 'T':
		show_image = True
		print("Show image output.\n")
	elif sys.argv[5] == 'F':
		show_image = False
		print("Do not show image output.\n")
	else:
		print("The fifth input is wrong. Please type the letter T or F, for True or False accordingly.")
		exit()

	# Call main
	main(unique_graph_name, detect3D, filtering_window, show_qsr, show_image)


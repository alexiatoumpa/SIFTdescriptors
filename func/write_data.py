#
# Copyright (C) 2019 University of Leeds
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

import csv

def WriteInFile(filename = file, delimiter = ',', data = []):
	with open(filename, mode='a') as csvfile:
		datawrite = csv.writer(csvfile, delimiter = delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
		datawrite.writerow(data)

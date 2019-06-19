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


# Inital settings of the code
def init():
    # Threshold value for distinguishing between 'simple' and 'complex' objects
	global MAX_thres_area
	MAX_thres_area = 4.0
	global how_many_max_sections
	global divider

	global sensitivity
	global h
	h = 1

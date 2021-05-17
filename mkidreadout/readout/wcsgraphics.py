"""
Author:	Katie Ayala
Date:	April 19, 2021

Classes for drawing astronomy shapes using a FITS ImageHDU WCS object 
to overlay on a real-time display image in Dashboard

Note: The display image QPixmap parent item origin (point [0,0] of the image 
	  data array) is in the upper left corner of the display axes. All pixel 
	  positions are in the parent's coordinates, and are in units of monitor 
	  screen pixels.
"""
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
# from astropy.coordinates import SkyCoord, Angle, angle_utilities 
from astropy.wcs import WCS
import astropy.units as units

__all__ = ['Compass', 'Scalebar']


class Compass(QGraphicsItemGroup):
	def __init__(self, scale, wcs=None, loc=(60,60), radius=40, color='lime', linewidth=1, parent=None):
		"""
		This class draws a WCS compass.
		
		INPUTS:
			scale - scale of display image from dashboard configuration file. The value is unitless.
			wcs - WCS object from display image FITS header
			loc - pixel position as an (x,y) coordinate pair of the compass vertex in parent coordinates
			radius - compass arm length in pixels 
			color - color of compass lines and labels
			linewidth - linewidth of compass arrows in pixels 
			parent - QPixmap object of live display image
		"""		
		super(Compass, self).__init__(parent=parent)
		self.scale = scale
		self.loc = loc
		self.radius = radius
		self.color = color
		self.linewidth = linewidth

		# self.wcs = wcs if wcs is not None else celestial_frame_to_wcs(ICRS())
		w = WCS(naxis=2)
		w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
		self.wcs = w

		### Initialize graphics with null lines and transparent labels
		pixelpts = self._pixelpoints(loc=self.loc, wcs=self.wcs, scale=self.scale, radius=0.0)
		q_lines_N = self._draw_line(p1=pixelpts[0], p2=pixelpts[1], arrowhead_length=0.0)
		q_lines_E = self._draw_line(p1=pixelpts[0], p2=pixelpts[2], arrowhead_length=0.0)
		
		self.graphicsN = [QGraphicsLineItem(i) for i in q_lines_N]
		self.graphicsE = [QGraphicsLineItem(i) for i in q_lines_E]
		for item in self.graphicsN + self.graphicsE:			
			self.addToGroup(item) 

		self.label_N = QGraphicsSimpleTextItem("N")
		self.label_E = QGraphicsSimpleTextItem("E")
		self.addToGroup(self.label_N)
		self.addToGroup(self.label_E)

		self.setVisible(False) 

	@staticmethod
	def _pixelpoints(loc, wcs, scale, radius, dir_offset=1.0e-4):
		"""
		Determine the pixel positions for the compass

		INPUTS:
			dir_offset - directional offset in degrees
		"""
		x, y = np.array(loc) / scale
		coord = SkyCoord.from_pixel(x, y, wcs)
		coordN = SkyCoord(ra=coord.ra.deg, dec=coord.dec.deg + dir_offset, unit=units.deg)
		coordE = SkyCoord(ra=coord.ra.deg + dir_offset, dec=coord.dec.deg, unit=units.deg)
		xe, ye = coordE.to_pixel(wcs)
		xn, yn = coordN.to_pixel(wcs)

		q_points = np.array([QPointF(x,y), QPointF(xn,yn), QPointF(xe,ye)]) * scale
		
		### Calculate points in the (xn,yn) or (xe,ye) direction a given distance from (x,y)
		def _offset_xy(p1, p2, dist):
			d = np.sqrt(np.abs(p2.x() - p1.x())**2 + np.abs(p2.y() - p1.y())**2)
			x = p1.x() + ((dist/d) * (p2.x() - p1.x()))
			y = p1.y() + ((dist/d) * (p2.y() - p1.y()))
			return QPointF(x,y)
		q_points[1] = _offset_xy(q_points[0], q_points[1], radius)
		q_points[2] = _offset_xy(q_points[0], q_points[2], radius)

		return q_points

	@staticmethod
	def _draw_line(p1, p2, arrowhead_length=6.0, arrowhead_angle=30.0):
		"""
		Create the compass arrow lines as QLineF objects

		INPUTS:
			p1 - a QPointF object (compass vertex position)
			p2 - a QPointF object
			arrowhead_length - length of arrowhead lines in pixels
			arrowhead_angle - angle of arrowhead lines in degrees
		"""
		compassline = QLineF(p1, p2)
		endpt = compassline.p2()

		### Create arrowhead lines
		arrowline1 = QLineF(endpt, p1)
		arrowline1.setAngle(arrowline1.angle() + arrowhead_angle)
		arrowline1.setLength(arrowhead_length)
		arrowline2 = QLineF(endpt, p1)
		arrowline2.setAngle(arrowline2.angle() - arrowhead_angle)
		arrowline2.setLength(arrowhead_length)

		return [compassline, arrowline1, arrowline2]

	@staticmethod
	def _update_label(q_text, loc, q_brush, pointsize):
		"""
		Update the label position and set the font and color 

		INPUTS:
			q_text - a QGraphicsTextItem (label text)
			loc - label pixel position as a QPointF object
			pointsize - point size of the font. The size of the font is device independent.
			q_brush - a QBrush object to set the text fill color
		"""
		offset = q_text.boundingRect().center()
		q_text.setPos(loc.x() - offset.x(), loc.y() + offset.y()/2.0)
		
		### Set the text fill color
		q_text.setBrush(q_brush)
		### Set the font for the label text
		font = QFont()
		font.setPointSize(pointsize)
		q_text.setFont(font)

	def update_wcs(self, new_wcs):
		""" 
		Use the WCS object to update the positions of the compass lines
		"""
		pixelpts = self._pixelpoints(loc=self.loc, wcs=new_wcs, scale=self.scale, radius=self.radius)
		q_lines_N = self._draw_line(p1=pixelpts[0], p2=pixelpts[1], arrowhead_length=self.radius/6.0)
		q_lines_E = self._draw_line(p1=pixelpts[0], p2=pixelpts[2], arrowhead_length=self.radius/6.0)

		for i in range(0,3):
			self.graphicsN[i].setLine(q_lines_N[i])
			self.graphicsE[i].setLine(q_lines_E[i])

		### Set up the QPen for drawing the graphics
		q_pen = QPen()
		q_color = QColor(self.color)
		q_brush = QBrush(q_color)
		q_pen.setBrush(q_brush)
		q_pen.setWidth(self.linewidth)

		for item in self.graphicsN + self.graphicsE:
			item.setPen(q_pen)

		### Update the "N" and "E" arrow labels
		self._update_label(self.label_N, pixelpts[1], q_brush, 11)
		self._update_label(self.label_E, pixelpts[2], q_brush, 11)


class Scalebar(QGraphicsItemGroup):
	def __init__(self, scale, wcs=None, loc=(40,840), length=0.1, length_unit='arcsecond', 
				 color='lime', linewidth=2, parent=None):
		"""
		This class draws a WCS scale bar.

		INPUTS:
			scale - scale of display image from dashboard configuration file. The value is unitless.
			wcs - WCS object from display image FITS header
			loc - pixel position as an (x,y) coordinate pair of the scale bar in parent coordinates
			length - angular size of scale bar in units of length_unit
			length_unit - unit of scale bar length ['arcsecond', 'milliarcsecond', 'degree']
			color - color of scale bar and label
			linewidth - linewidth of scale bar in pixels 
			parent - QPixmap object of live display image			
		"""
		super(Scalebar, self).__init__(parent=parent)
		self.scale = scale
		self.loc = loc
		self.length = length
		self.length_unit = length_unit
		self.color = color
		self.linewidth = linewidth

		# self.wcs = wcs if wcs is not None else celestial_frame_to_wcs(ICRS())
		w = WCS(naxis=2)
		w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
		self.wcs = w

		### Initialize graphics with null lines
		pixelpts = self._pixelpoints(loc=self.loc, wcs=self.wcs, scale=self.scale, length=0.0, length_unit=self.length_unit)
		self.graphics = QGraphicsLineItem(QLineF(pixelpts[0], pixelpts[1]))
		self.addToGroup(self.graphics) 

		### Initialize label indicating the angular size
		sep = Angle(self.length, unit=units.Unit(self.length_unit))
		if self.length_unit == 'arcsecond':
			text = sep.to_string(unit=units.arcsecond, decimal=True, precision=1) + u'\N{DOUBLE PRIME}'		
		elif self.length_unit == 'milliarcsecond':
			text = sep.to_string(unit=units.milliarcsecond, decimal=True) + ' mas'		
		elif self.length_unit == 'degree':
			text = sep.to_string(unit=units.degree, decimal=True, pad=True) + u'\N{DEGREE SIGN}'	

		self.label = QGraphicsSimpleTextItem(text)
		self.addToGroup(self.label)

		self.setVisible(False) 

	@staticmethod
	def _pixelpoints(loc, wcs, scale, length, length_unit, dir_offset=1.0):
		"""
		Determine the pixel positions for the scale bar endpoints

		INPUTS:
			dir_offset - directional offset in pixels for reference coordinate
		"""
		x1, y1 = np.array(loc) / scale
		coord1 = SkyCoord.from_pixel(x1, y1, wcs)
		### Reference coord with offset in positive x-direction
		ref_coord = SkyCoord.from_pixel(x1 + dir_offset, y1, wcs)

		### On-sky position angle (E of N) between coord1 and ref_coord
		pa = coord1.position_angle(ref_coord)
		sep = Angle(length, unit=units.Unit(length_unit))
		# coord2 = coord1.directional_offset_by(pa,sep)

		### Compute the point with the given offset from the input coord
		def _offset_by(coord, posang, distance):
			cos_a, sin_a = np.cos(distance), np.sin(distance)
			cos_c, sin_c = np.sin(coord.dec), np.cos(coord.dec)
			cos_B, sin_B = np.cos(posang), np.sin(posang)

			cos_b = cos_c * cos_a + sin_c * sin_a * cos_B
			xsin_A = sin_a * sin_B * sin_c
			xcos_A = cos_a - cos_b * cos_c

			A = Angle(np.arctan2(xsin_A, xcos_A), units.radian)
			new_ra = (Angle(coord.ra, units.radian) + A).wrap_at(360.0*units.deg).to(units.deg)
			new_dec = Angle(np.arcsin(cos_b), units.radian).to(units.deg)
			return SkyCoord(new_ra, new_dec, frame=coord.frame)

		coord2 = _offset_by(coord1, pa, sep)
		x2, y2 = coord2.to_pixel(wcs)
	
		q_points = np.array([QPointF(x1,y1), QPointF(x2,y2)]) * scale
		return q_points

	@staticmethod
	def _update_label(q_text, loc, q_brush, pointsize):
		"""
		Update the label position and set the font and color 

		INPUTS:
			q_text - a QGraphicsTextItem (label text)
			loc - label pixel position as a QPointF object
			pointsize - point size of the font. The size of the font is device independent.
			q_brush - a QBrush object to set the text fill color
		"""
		offset = q_text.boundingRect().center()
		q_text.setPos(loc.x() - offset.x(), loc.y() + offset.y()/2.0)

		### Set the text fill color
		q_text.setBrush(q_brush)
		### Set the font for the label text
		font = QFont()
		font.setPointSize(pointsize)
		q_text.setFont(font)

	def update_wcs(self, new_wcs):
		""" 
		Use the WCS object to update the positions of scale bar endpoints
		"""
		pixelpts = self._pixelpoints(loc=self.loc, wcs=new_wcs, scale=self.scale, length=self.length, length_unit=self.length_unit)
		line = QLineF(pixelpts[0], pixelpts[1])
		midpt = line.pointAt(0.5)
		self.graphics.setLine(line)

		### Set up the QPen for drawing the graphics
		q_pen = QPen()
		q_color = QColor(self.color)
		q_brush = QBrush(q_color)
		q_pen.setBrush(q_brush)
		q_pen.setWidth(self.linewidth)

		self.graphics.setPen(q_pen)

		### Update label
		self._update_label(self.label, midpt, q_brush, 11)
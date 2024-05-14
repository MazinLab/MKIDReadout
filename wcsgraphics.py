class Compass(QGraphicsItemGroup):
	def __init__(self, scale=0.01, wcs=None, loc=(60,60), radius=5.0e-2, parent=None):
		"""
		Note: The image origin (point (0,0) of the data array) is in the upper left 
			  corner of the axes

		INPUTS:
			scale - scale of display image
			wcs - WCS object from display image FITS header
			loc - compass vertex location [x,y] coordinate pair
			radius - compass arm length as a fraction of the smallest axis size 
		"""		
		super(Compas, self).__init__(parent=parent)
		self.scale = scale
		self.wcs = wcs
		self.loc = loc
		self.radius = radius * scale * min(wcs.pixel_shape)

        #Create all graphics once at init, update their positions and visibility later
        pixelpts = self._pixelpoints(self.loc, self.wcs, self.scale)
        self.graphicsN = self._draw_line(self.radius, pixelpts[0], pixelpts[1], text="N")
        self.graphicsE = self._draw_line(self.radius, pixelpts[0], pixelpts[2], text="E")

        for item in self.graphicsN+self.graphicsE:
            self.addToGroup(item)  #NB I'm surpried qt doesn't offer something like addToGroup(listOfItems)
        self.setVisible(False)  # NB visibility setting isn't "creating"

    @staticmethod
    def _pixelpoints(loc, wcs, scale, dir_offset=1.0e-4):
        """ Just determine the pixel positions for the compass"""
        x, y = np.array(loc) / scale
        coord = SkyCoord.from_pixel(x, y, wcs)
        coordN = SkyCoord(ra=coord.ra.deg, dec=coord.dec.deg + dir_offset, unit=units.deg)
        coordE = SkyCoord(ra=coord.ra.deg + dir_offset, dec=coord.dec.deg, unit=units.deg)
        xe, ye = coordE.to_pixel(wcs)
        xn, yn = coordN.to_pixel(wcs)
        return np.array([np.array([x,y]), np.array([xn,yn]), np.array([xe,ye])])  * scale

    @staticmethod
    def _draw_line(radius, p1, p2, text, arrowhead_angle=30, color='lime', linewidth=1):
        """
        This function draws an arrow with an appropriate label showing the increasing
        direction of the coordinate (East or North)

        INPUTS:
            radius - ???
            p1 - an [x,y] coordinate pair (compass vertex)
            p2 - an [x,y] coordinate pair
            text - label associated with the arrow ("N" or "E")
            arrowhead_angle - angle of arrowhead lines
        """
        compassline = QLineF(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))
        compassline.setLength(radius)
        endpt = compassline.pointAt(1)
        arrowhead_length = radius / 6.0

        ### Draw arrowhead lines
        arrowline1 = QLineF(endpt, QPointF(p1[0], p1[1]))
        arrowline1.setAngle(arrowline1.angle() + arrowhead_angle)
        arrowline1.setLength(arrowhead_length)
        
        arrowline2 = QLineF(endpt, QPointF(p1[0], p1[1]))
        arrowline2.setAngle(arrowline2.angle() - arrowhead_angle)
        arrowline2.setLength(arrowhead_length)

        ### Set up the QPen for drawing the compass arrow lines
        q_color = QColor()
        q_color.setNamedColor(color)
        q_pen = QPen(q_color)
        q_pen.setWidth(linewidth)

        compassline = QGraphicsLineItem(compassline)
        arrowline1 = QGraphicsLineItem(arrowline1)
        arrowline2 = QGraphicsLineItem(arrowline2)
        compassline.setPen(q_pen)
        arrowline1.setPen(q_pen)
        arrowline2.setPen(q_pen)

        ### Draw label
        font = QFont()
        font.setPointSize(11)

        arrowlabel = QGraphicsTextItem(text)
        arrowlabel.setFont(font)
        arrowlabel.setDefaultTextColor(q_color)
        offset = arrowlabel.boundingRect().center()
        arrowlabel.translate(endpt.x() - offset.x(), endpt.y())

        return [compassline, arrowline1, arrowline2, arrowlabel]

    def update_wcs(self, wcs):
        """ Use the wcs object to update the positions of the compass lines """
        pixelpts = self._pixelpoints(self.loc, wcs, self.scale)
        #TODO Update self.graphicsN[
        for line in self.graphicsN[:-1]:  # NB [compassline, arrowline1, arrowline2, arrowlabel]
            #TODO determine new points and update points
            line.setLine(x1, y1, x2, y2)  # NB or some such!

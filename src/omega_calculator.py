import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

class OmegaCalculator:
    def __init__(self):
        """
        Instantiates an instance of the OmegaCalculator Class
        """
        self.original_cone_pos = None
        self.cone_pos = None
        self.original_tel_pos = None
        self.tel_pos = None
        self.r_cone = None
        self.r_tel_core = None
        self.r_tel_reach = None

    def update(self, cone_pos = (0, 0), tel_pos = (0, 0), r_cone = None, r_tel_core = None, r_tel_reach = None):
        """
        Sets parameters of the Cherenkov Cone and the Telescope in a 2D-plane
        
        Parameters
        ----------
        cone_pos: (float, float)
            Default: (0, 0)
            Position of centre of cherenkov shower cone in 2D plane
            Original position will be saved, but class will treat cone_pos as centre of axis (0, 0)

        tel_pos: (float, float)
            Default: (0, 0)
            Position of centre of telescope in 2D plane
            Original position will be saved, but class will offset tel_pos by cone_pos to align axes for calculations

        r_cone: float
            Radius of cherenkov shower cone

        r_tel_core: float
            Radius of telescope inner circle (core)

        r_tel_reach: float
            Radius of telescope outer circle (reach)
        """
        self.original_cone_pos = cone_pos # Used for plotting
        self.cone_pos = (0, 0) # Always assume cone is at origin
        self.original_tel_pos = tel_pos # Used for plotting
        self.tel_pos = (tel_pos[0] - cone_pos[0], tel_pos[1] - cone_pos[1]) # Offset Telescope by cone position if not at origin
        self.r_cone = r_cone
        self.r_tel_core = r_tel_core
        self.r_tel_reach = r_tel_reach

    def plotArc(start, end, radius, centre, dw, num_ele):
        """
        Plots an arc between start and end with the specified angle.

        Parameters:
        -----------
        start: tuple
            Starting point (x1, y1)

        end: tuple
            Last point (x2, y2)

        angle: float
            Angle (in degrees) defining the arc curvature.

        radius: float
            Radius of the arc.

        centre: tuple
            Coordinates for the centre of the arc.

        dw: float
            The angle δω in radians.

        num_ele: int
            Number of points to generate on the arc.
        """
        # Distance between points
        d = np.sqrt((end[0] - start[0])**2 + (end[1]-start[1])**2)

        # Start and end angles of arc
        start_angle = np.arctan2(start[1] - centre[1], start[0] - centre[0])
        end_angle = np.arctan2(end[1] - centre[1], end[0] - centre[0])

        # Check if dw is reflex angle, create corresponding angle linspace from start angle to end angle
        # If dw not reflex angle (<= 180 deg), plot cw
        if dw <= np.pi:
            theta_values = np.linspace(start_angle, end_angle, num_ele)
        # If dw reflex angle, plot ccw
        else:
            theta_values = np.linspace(start_angle, start_angle + dw, num_ele)

        # Generate arc points
        arc_x = centre[0] + radius * np.cos(theta_values)
        arc_y = centre[1] + radius * np.sin(theta_values)

        # Plot arc
        plt.plot(arc_x, arc_y, color = "magenta", label = "δω")

        # Plot given points
        plt.scatter(*start, color = 'green', label = 'Intersection Point 1')
        plt.scatter(*end, color = 'darkseagreen', label = 'Intersection Point 2')
        plt.text((start[0] + end[0])/2, (start[1] + end[1])/2 + 0.15, "δω")

    def graphCircles(self, plot_variables = False, dw = [], start = [], end = [], num_ele = 100):
        """
        Plots the Cherenkov cone, telescope core and telescope's reach and other optional variables such as ρ and δω.

        Parameters:
        -----------
        cone_pos: tuple
            Coordinates for the centre of the Chrenkov cone.

        tel_pos: tuple 
            Coordinates for the centre of the telescope.

        r_cone: float
            Radius of the Cherenkov cone at the telescope.

        r_tel_core: float
            Radius of the telescope core.

        r_tel_reach: float
            Radius of the reach of the telescope.

        plot_variables: boolean
            Controls whether to plot the other variables (ρ and δω)(if True) or not (if False).

        dw: numpy array
            Array of the δω angles in degrees.

        start: tuple, list, numpy array
            Coordinates for the starting intersection point between the Cherenkov cone and the telescope's core and/or reach.
            If there are 2 "δω"s, put the coordinates of the two starting intersection points in an array within an array as follows: 
            ( (x_s1, y_s1), (x_s2, y_s2) )

        end: tuple, list, numpy array
            Coordinates for the ending intersection point between the Cherenkov cone and the telescope's core and/or reach.
            If there are 2 "δω"s, put the coordinates of the two ending intersection points in an array within an array as follows: 
            ( (x_e1, y_e1), (x_e2, y_e2) )

        num_ele: int
            Number of points to generate on the circles and δω.
        """
        cone_pos = self.original_cone_pos
        tel_pos = self.original_tel_pos
        
        # Define circle
        theta = np.linspace(0, 2*np.pi, num_ele)
        x = np.cos(theta)
        y = np.sin(theta)

        plt.figure(figsize = (6, 6))
        
        # Plot Cherenkov Cone
        plt.plot(cone_pos[0] + x*self.r_cone, cone_pos[1] + y*self.r_cone, label = "Shower Cone", linestyle = "-", color = "black")

        # Plot Telescope Core and Reach
        plt.plot(tel_pos[0] + x*self.r_tel_core, tel_pos[1] + y*self.r_tel_core, label = 'Telescope Core', linestyle = '--', color = 'red')
        plt.plot(tel_pos[0] + x*self.r_tel_reach, tel_pos[1] + y*self.r_tel_reach, label = 'Telescope Reach', linestyle = '--', color = 'blue')

        # Labels and title
        plt.title("Shower Cone and Telescope Projection")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)

        # Aspect ratio set such that circles always look correct
        plt.gca().set_aspect("equal", adjustable="box")

        if plot_variables == True:
            # Create straight line indicating radius of cone (Rho) along X-axis
            rho_x = [cone_pos[0], cone_pos[0] + self.r_cone]
            rho_y = [cone_pos[1], cone_pos[1]]
            plt.plot(rho_x, rho_y, label = "ρ", linestyle = "-", color = "green")
            plt.text((cone_pos[0] + cone_pos[0] + self.r_cone)/2, cone_pos[1] + 0.15, "ρ")

            # Depending on dw, plot 1 or 2 arc between intersection points
            if len(dw) == 1:
                self.plotArc(start, end, self.r_cone, cone_pos, dw[0], num_ele)
            elif len(dw) == 2:
                self.plotArc(start[0], end[0], self.r_cone, cone_pos, dw[0], num_ele)
                self.plotArc(start[1], end[1], self.r_cone, cone_pos, dw[1], num_ele)

    def getIntersectionPoints(self, r_tel):
        """
        Returns 2 points in the 2D-plane that correspond to where the two circles intersect
        
        Parameters
        -----------
        r_tel: float
            Radius of either the core or reach circle of the telescope
            Needs to be passed in as otherwise function doesn't know which one to calculate for
        """
        # If q is a quantity, convert to unitless value
        def to_consistent_units(q):
            if isinstance(q, u.Quantity):
                return q.to(u.m).value
            else:
                return q

        # Define constants
        r = to_consistent_units(self.r_cone)
        R = to_consistent_units(r_tel)
        L = to_consistent_units(self.tel_pos[0])
        h = to_consistent_units(self.tel_pos[1])
        K = R**2 - r**2 - L**2 - h**2

        a = (4*(h**2 + L**2))
        b = (4*K*h)
        c = ((K**2) - 4*(L**2)*(r**2))

        # Calculate roots
        y = np.roots([a, b, c])
        
        x = [None, None]
        # Determine corresponding x value for each y value
        for idx, y_val in enumerate(y):   
            cone_xs = [np.sqrt(r**2 - y_val**2), -np.sqrt(r**2 - y_val**2)]
            tel_xs = [np.sqrt(R**2 - (y_val-h)**2) + L, -np.sqrt(R**2 - (y_val-h)**2) + L]

            # If approximately same x point produced in cone_xs and tel_xs, must be correct x point
            match = next((val for val in cone_xs if any(np.isclose(val, a) for a in tel_xs)), None)
            if match is not None:
                x[idx] = match

        return (x[0], y[0]), (x[1], y[1])

    def getClosestPoint(self, point, pair):
        """
        Determines which of the points in the pair is closest to the reference point

        Parameters
        -----------
        point: (float, float)
            Reference point, usually an intersection point where the cone intersects the core of the telescope
        
        pair: ((float, float), (float, float))
            Comparison points, usually the intersection points where the cone intersects the annulus of the telescope   
        """
        getDistance = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # Calculates distance between points
        p0_dist = getDistance(point, pair[0])
        p1_dist = getDistance(point, pair[1])

        # If pair[0] point is closer, return pair[0] point
        if p0_dist < p1_dist:
            return pair[0]
        
        # If pair[1] point is closer, return pair[1] point
        elif p0_dist > p1_dist:
            return pair[1]
        
        # If pair[0] = pair[1], equal distance. Should not happen
        else:
            raise ValueError("Equal distance between reference point and pair. Unable to determine closest point")
            return (0, 0)

    def determineArcType(self):
        """
        Determines if the omega is the major or minor arc depending on position and radius of telescope
        Approximation, doesn't always work as intended. For example if centre of cone is just barely inside/outside radius of telescope
        """
        # Calculate distance
        dist = np.hypot(0 - self.tel_pos[0], 0 - self.tel_pos[1])

        # If centre of cone within telescope annulus but not telescope core, find major angle by default
        if self.r_tel_core <= dist <= self.r_tel_reach:
            return "major"
        # Otherwise find minor angle
        else:
            return "minor"

    def calculateOmegaFromPoints(self, point1, point2, arcType):
        """
        Calculates omega from between 2 points

        Parameters:
        -----------
        point1: (float, float)
            First intersection point

        point2: (float, float)
            Second intersection point

        arcType: string
            Determines if omega returned is the major or minor angle between the two points
            Depending on position of telescope and radius, either could be best for the situation

        degrees: bool
            Determines whether returned angle is in degrees or radians
            by default returns degrees
        """
        # Calculate angles at both points
        angle1 = np.arctan2(point1[1], point1[0])
        angle2 = np.arctan2(point2[1], point2[0])

        # Find angle difference, default to minor angle
        minor_arc = (angle2 - angle1) % (2*np.pi)
        minor_arc = min(minor_arc, (2*np.pi) - minor_arc)

        # If major requested return major angle, otherwise return minor angle
        if arcType == "major":
            return (2*np.pi) - minor_arc
        else:
            return minor_arc

    def getOmegas(self):
        """
        Returns omega, the angle of the cone that is within the telescope that can be picked up by the mirror
        Different cases are checked, and the corresponding angle is given
        """
        # Define parameters of intersecting circles
        dist = np.sqrt(self.tel_pos[0]**2 + self.tel_pos[1]**2)
        
        outside_tel_reach = dist >= self.r_cone + self.r_tel_reach
        inside_tel_reach = dist <= self.r_tel_reach - self.r_cone
        intersect_tel_reach = ~outside_tel_reach and ~inside_tel_reach

        outside_tel_core = dist >= self.r_cone + self.r_tel_core
        inside_tel_core = dist <= self.r_tel_core - self.r_cone
        intersect_tel_core = ~outside_tel_core and ~inside_tel_core

        tel_reach_inside_cone = dist <= self.r_cone - self.r_tel_reach
        tel_core_inside_cone = dist <= self.r_cone - self.r_tel_core

        # Case 1 - Cone fully outside of Telescope or only one intersection point
        if (outside_tel_reach):
            return 0.0
        
        # Case 2 - Cone fully inside of Telescope Annulus (or only one intersection) but fully outside of Telescope Core (or only one intersection)
        elif (inside_tel_reach and outside_tel_core):
            return 360.0
        
        # Case 3 - Cone fully inside of Telescope Core (or only one intersection)
        elif (inside_tel_core):
            return 0.0
        
        # Case 4 - Cone fully inside of Telescope Annulus (or only one intersection), and fully surrounding Telescope core (or only one intersection)
        elif (inside_tel_reach and tel_core_inside_cone):
            return 360.0
        
        # Case 5 - Cone fully surrounding Telescope Annulus (or only one intersection)
        elif (tel_reach_inside_cone):
            return 0.0
        
        # Case 6 - Cone intersecting twice solely with Telescope Annulus
        elif (intersect_tel_reach and not intersect_tel_core):
            points = self.getIntersectionPoints(self.r_tel_reach)
            omega = self.calculateOmegaFromPoints(points[0], points[1], self.determineArcType())
            return omega * 180/np.pi
        
        # Case 7 - Cone intersecting twice solely with Telescope Core
        elif (intersect_tel_core and not intersect_tel_reach):
            points = self.getIntersectionPoints(self.r_tel_core)
            omega = self.calculateOmegaFromPoints(points[0], points[1], self.determineArcType())
            return omega * 180/np.pi
        
        # Case 8 - Cone intersecting with both Telescope Annulus and Core
        elif (intersect_tel_reach and intersect_tel_core):
            reach_pair = self.getIntersectionPoints(self.r_tel_reach)
            core_pair = self.getIntersectionPoints(self.r_tel_core)

            # Use core_pair as reference, compare to reach_pair to find corresponding pairs
            pair1 = (core_pair[0], self.getClosestPoint(core_pair[0], reach_pair))
            pair2 = (core_pair[1], self.getClosestPoint(core_pair[1], reach_pair))

            # Calculate both omegas
            omega1 = self.calculateOmegaFromPoints(pair1[0], pair1[1], "minor")
            omega2 = self.calculateOmegaFromPoints(pair2[0], pair2[1], "minor")

            # omega1 should equal omega2
            if not np.isclose(omega1, omega2, rtol=1e-5):
                print("Error: Omega 1, ", omega1, " != Omega 2, ", omega2, ". Returning Omega 1")

            # Return omega 1
            return omega1 * 180/np.pi
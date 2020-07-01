## Assessment solution for baseball detection and tracking


After exploring various options, the solution I settled on is a combination of CSRT tracking and dynamic modelling to correctly track the balls location, while using Hough circles to identify the centre points and radiuses. 
The final centre points and radiuses are stored in a txt file called “circles.txt” in the form

    x,y,radius

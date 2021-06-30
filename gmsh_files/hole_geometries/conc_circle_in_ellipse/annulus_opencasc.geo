SetFactory("OpenCASCADE");
n=50;
a=1; b=0.75;
r_i=0.5;

c1 = newl; Circle(c1) = {0, 0, 0, r_i, 0, 2*Pi};

p0 = newp; Point(p0) = {0, 0, 0}; //center
p1 = newp; Point(p1) = {a, 0, 0}; //point
p2 = newp; Point(p2) = {0, b, 0}; //point
p3 = newp; Point(p3) = {-a, 0, 0};//point
p4 = newp; Point(p4) = {0, -b, 0}; //point

e1 = newl; Ellipse(e1) = {p1, p0, p2}; //ellipse arc
e2 = newl; Ellipse(e2) = {p2, p0, p3}; //ellipse arc
e3 = newl; Ellipse(e3) = {p3, p0, p4}; //ellipse arc
e4 = newl; Ellipse(e4) = {p4, p0, p1}; //ellipse arc


Curve Loop(1) = {c1};
Curve Loop(2) = {e1,e2,e3,e4};
Plane Surface(1) = {1,2};
Transfinite Line {c1} = n;
Transfinite Line {e1,e2,e3,e4} = 0.25*n*a/r_i;
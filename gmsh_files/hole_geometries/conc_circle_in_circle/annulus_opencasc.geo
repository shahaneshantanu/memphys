SetFactory("OpenCASCADE");
n=195;
r_o=1;
r_i=0.5;
Circle(1) = {0, 0, 0, r_i, 0, 2*Pi};
Circle(2) = {0, 0, 0, r_o, 0, 2*Pi};
Curve Loop(1) = {1};
Curve Loop(2) = {2};
Plane Surface(1) = {1,2};
Transfinite Line {1} = n;
Transfinite Line {2} = n*r_o/r_i;
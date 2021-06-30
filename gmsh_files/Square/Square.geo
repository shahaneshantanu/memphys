boxdim = 1.0; 
n_points=40; //no of points on each edge

Point(1) = {0,0,0};
Point(2) = {boxdim,0,0};
Point(3) = {boxdim,boxdim,0};
Point(4) = {0,boxdim,0};

Line(101) = {1,2};
Line(102) = {2,3};
Line(103) = {3,4};
Line(104) = {4,1};

Transfinite Line {101:104} = n_points;

Line Loop(1001) = {101,102,103,104}; Plane Surface(1002) = {1001};

// Transfinite Surface "*"; //uncomment this for structured grid 
// Recombine Surface "*";
// Transfinite Volume "*"; //uncomment this for structured grid 
// Coherence;Coherence;

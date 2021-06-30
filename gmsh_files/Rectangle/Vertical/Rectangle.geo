lx = 1; ly=10*lx; 
nx=5; 

Point(1) = {0,0,0};
Point(2) = {lx,0,0};
Point(3) = {lx,ly,0};
Point(4) = {0,ly,0};

Line(101) = {1,2};
Line(102) = {2,3};
Line(103) = {3,4};
Line(104) = {4,1};

Transfinite Line {102,104} = nx*ly/lx;
Transfinite Line {101,103} = nx;

Line Loop(1001) = {101,102,103,104}; Plane Surface(1002) = {1001};

// Transfinite Surface "*"; //uncomment this for structured grid 
// Recombine Surface "*";
// Transfinite Volume "*"; //uncomment this for structured grid 
// Coherence;Coherence;

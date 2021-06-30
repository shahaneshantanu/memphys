boxdim = 1; 
n_points=53; //no of points on each edge

Point(1) = {0,0,0};
Point(2) = {boxdim,0,0};
Point(3) = {boxdim,boxdim,0};
Point(4) = {0,boxdim,0};
Point(5) = {0,0,boxdim};
Point(6) = {boxdim,0,boxdim};
Point(7) = {boxdim,boxdim,boxdim};
Point(8) = {0,boxdim,boxdim};


Line(20) = {1,2};
Line(21) = {2,3};
Line(22) = {3,4};
Line(23) = {4,1};
Line(24) = {5,6};
Line(25) = {6,7};
Line(26) = {7,8};
Line(27) = {8,5};
Line(28) = {2,6};
Line(29) = {3,7};
Line(30) = {4,8};
Line(31) = {1,5};

Transfinite Line {20:31} = n_points;


Line Loop(51) = {20,21,22,23}; Plane Surface(101) = {51};
Line Loop(52) = {24,25,26,27}; Plane Surface(102) = {52};
Line Loop(53) = {-23,30,27,-31}; Plane Surface(103) = {53};
Line Loop(54) = {21,29,-25,-28}; Plane Surface(104) = {54};
Line Loop(55) = {-20,31,24,-28}; Plane Surface(105) = {55};
Line Loop(56) = {22,30,-26,-29}; Plane Surface(106) = {56};

Surface Loop(1001) = {101, 102, 103, 104, 105, 106}; Volume(1002) = {1001};

// Transfinite Surface "*"; //uncomment this for structured grid 
// Recombine Surface "*";
// Transfinite Volume "*"; //uncomment this for structured grid 
// Coherence;Coherence;
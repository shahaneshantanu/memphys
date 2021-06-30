//DO NOT USE THIS GEO FILE: 
//adds 2 chords along diagonal with points (on line elements) falsely identifed as boundaries
n=10;
r_o=1;
r_i=0.5;

Point(1) = {0,0,0};
Point(2) = {r_o,0,0};
Point(3) = {0,r_o,0};
Point(5) = {-r_o,0,0};
Point(6) = {0,-r_o,0};

Circle(1) = {2,1,3};
Circle(2) = {3,1,5};
Circle(3) = {5,1,6};
Circle(4) = {6,1,2};

// Line Loop(1) = {1,2,3,4};

Point(200) = {r_i,0,0};
Point(300) = {0,r_i,0};
Point(500) = {-r_i,0,0};
Point(600) = {0,-r_i,0};

Circle(100) = {200,1,300};
Circle(200) = {300,1,500};
Circle(300) = {500,1,600};
Circle(400) = {600,1,200};

// Line(5) = {200,2}; Line(6) = {300,3};
// Line(7) = {500,5}; Line(8) = {600,6};
// Line Loop(1) = {5,1,-6,-100};
// Line Loop(2) = {6,2,-7,-200};
// Line Loop(3) = {7,3,-8,-300};
// Line Loop(4) = {8,4,-5,-400};

// Plane Surface(1) = {1}; Plane Surface(2) = {2};
// Plane Surface(3) = {3}; Plane Surface(4) = {4};

Line(5) = {200,2}; Line(6) = {500,5};
Line Loop(1) = {5,1,2,-6,-200,-100};
Line Loop(2) = {6,3,4,-5,-400,-300};

Plane Surface(1) = {1}; Plane Surface(2) = {2};

Transfinite Line {5,6} = n;
Transfinite Line {100,200,300,400} = 0.5*3.15*r_i*n/(r_o-r_i);
Transfinite Line {1,2,3,4} = 0.5*3.15*r_o*n/(r_o-r_i);

// Line Loop(100) = {100,200,300,400};

// // Annulus : The surface between the little circle and the large circle
// Plane Surface(1) = {1, 100};

// Transfinite Line {1:4} = n*r_o/r_i;
// Transfinite Line {100,200,300,400} = n;


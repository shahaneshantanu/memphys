SetFactory("OpenCASCADE");
n=40;
dia_x=1.0; dia_y=2.0;
lx=15*dia_y; ly=30*dia_x;

rect = newl; Rectangle(rect) = {-lx/3, -ly/2, 0, lx, ly, 0};


c0 = newp; Point(c0) = {0, 0, 0};
p1 = newp; Point(p1) = {dia_x, 0, 0};
p2 = newp; Point(p2) = {0, dia_y, 0};
p3 = newp; Point(p3) = {-dia_x, 0, 0};
p4 = newp; Point(p4) = {0, -dia_y, 0};

e1 = newl; Ellipse(e1) = {p1, c0, p2};
e2 = newl; Ellipse(e2) = {p2, c0, p3};
e3 = newl; Ellipse(e3) = {p3, c0, p4};
e4 = newl; Ellipse(e4) = {p4, c0, p1};
ell_loop=newl; Curve Loop(ell_loop) = {e1,e2,e3,e4};
ell_surf=news; Plane Surface(ell_surf) = {ell_loop};
final_surf=news; BooleanDifference(final_surf) = { Surface{rect}; Delete; }{ Surface{ell_surf}; Delete; };

Mesh.CharacteristicLengthMax=5*3.14*0.5*(dia_y+dia_x)/n;
Transfinite Line {e1,e2,e3,e4} = n/4;
SetFactory("OpenCASCADE");
n=25; refine_hole_factor=5;
circ_rad=0.04; ell_rad_maj=0.06; ell_rad_min=0.04;

p1=newp; Point(p1)={0,0,0};
p2=newp; Point(p2)={0.1,0,0};
p3=newp; Point(p3)={0.2,0.1,0};
pc1=newp; Point(pc1)={0.2,0.0,0};
p4=newp; Point(p4)={0.2,0.2,0};
p5=newp; Point(p5)={0.1,0.3,0};
p6=newp; Point(p6)={0,0.3,0};

Circle(2) = {p2,pc1,p3}; 
Line(1) = {p1,p2}; 
Line(3) = {p3,p4}; 
Line(4) = {p4,p5}; 
Line(5) = {p5,p6}; 
Line(6) = {p6,p1}; 
out_boundary=newll; Curve Loop(out_boundary) = {1,2,3,4,5,6};
out_surf = news; Surface(out_surf) = {out_boundary};

Circle(7) = {0.065, 0.1, 0, circ_rad, 0, 2*Pi};
circ_loop1=newll; Curve Loop(circ_loop1) = {7};
circ_surf1=news; Plane Surface(circ_surf1) = {circ_loop1};

Ellipse(8) = {0.09, 0.2, 0, ell_rad_maj, ell_rad_min};
ell_loop1=newl; Curve Loop(ell_loop1) = {8};
ell_surf1=news; Plane Surface(ell_surf1) = {ell_loop1};

final_surf1=news; BooleanDifference(final_surf1) = { Surface{out_surf}; Delete; }{ Surface{circ_surf1}; Delete; };
final_surf2=news; BooleanDifference(final_surf2) = { Surface{final_surf1}; Delete; }{ Surface{ell_surf1}; Delete; };

// Mesh.CharacteristicLengthMax=1E20;
Mesh.CharacteristicLengthMax=0.1/n;
Transfinite Line {7} = refine_hole_factor*(2*3.14*circ_rad/0.1)*n;
Transfinite Line {8} = refine_hole_factor*(2*3.14*Sqrt(0.5*(ell_rad_maj*ell_rad_maj + ell_rad_min*ell_rad_min))/0.1)*n;
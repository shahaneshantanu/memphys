SetFactory("OpenCASCADE");
nc=20; //points on circle
cof=5; //coarsening factor far-stream
diameter=1.0;
lx_up=20*diameter; lx_down=30*diameter; 
ly=20*diameter;

rect = newl; Rectangle(rect) = {-lx_up, -ly, 0, lx_down+lx_up, 2*ly, 0};

circ = newl; Circle(circ) = {0, 0, 0, 0.5*diameter, 0, 2*Pi};
circ_loop=newl; Curve Loop(circ_loop) = {circ};
circ_surf=news; Plane Surface(circ_surf) = {circ_loop};

final_surf=news; BooleanDifference(final_surf) = { Surface{rect}; Delete; }{ Surface{circ_surf}; Delete; };

Mesh.CharacteristicLengthMax=cof*3.14*diameter/nc;
Transfinite Line {circ} = nc;
// Field[1] = BoundaryLayer;
// Field[1].EdgesList = {circ};
// Field[1].thickness = 3*radius;
// Field[1].hwall_n = 2*3.14*radius/n; //Mesh Size Normal to the The Wall
// Field[1].hfar = Mesh.CharacteristicLengthMax;
// Field[1].ratio = 1.1;
// BoundaryLayer Field = 1;
//referene for boundary layers: http://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes
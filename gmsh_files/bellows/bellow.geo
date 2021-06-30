h=0.5; lambda=3; amp=0.3; lx=0;
n=25; nb=1;

Point(1) = {0,-h/2,0};
Point(2) = {lx,-h/2,0};
Point(3) = {lx,h/2,0};
Point(4) = {0,h/2,0};

Point(5) = {lambda*nb+lx,-h/2,0};
Point(6) = {lambda*nb+2*lx,-h/2,0};
Point(7) = {lambda*nb+2*lx,h/2,0};
Point(8) = {lambda*nb+lx,h/2,0};
Line(101) = {1,2}; Line(102) = {5,6};
Line(103) = {6,7}; Line(104) = {7,8};
Line(105) = {3,4}; Line(106) = {4,1};

nPoints = 100*nb; // Number of discretization points
plist_bottom[0] = 2; // First point label
For i In {1 : nPoints}
  x = lambda*nb*i/(nPoints+1); 
  y = amp*Cos(2*Pi*x/lambda) - 0.5*h - amp; 
  plist_bottom[i] = newp; Point(plist_bottom[i]) = {x+lx, y, 0};
EndFor
plist_bottom[nPoints+1] = 5; // Last point label
Spline(1) = plist_bottom[];

plist_top[0] = 3; // First point label
For i In {1 : nPoints}
  x = lambda*nb*i/(nPoints+1); 
  y = -amp*Cos(2*Pi*x/lambda) + 0.5*h + amp; 
  plist_top[i] = newp; Point(plist_top[i]) = {x+lx, y, 0};
EndFor
plist_top[nPoints+1] = 8; // Last point label
Spline(2) = plist_top[];

Line Loop(1000) = {101,1,102,103,104,-2,105,106};
Plane Surface(1001) = {1000};
Mesh.CharacteristicLengthMax=h/n;

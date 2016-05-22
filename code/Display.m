a = fopen('betafinal.mrc');
b = fread(a,[1,256], 'uint');
c = fread(a, 'float');
d = reshape(c,[180,180,180]);
d = d-min(c);
d = d/max(c);
normalized_Free_Energy_map = d;
x = 1:180;
y = 1:180;
z = 1:180;
figure
quantum=1.0/8;
isovalue=5*quantum;
surf1=isosurface(x,y,z,normalized_Free_Energy_map,isovalue);
p1 = patch(surf1);
isonormals(x,y,z,normalized_Free_Energy_map,p1);
set(p1,'FaceColor','red','EdgeColor','none','FaceAlpha',0.4); % set the color, mesh and transparency level of the surface
daspect([1,1,1])
view(3); axis tight
camlight; lighting gouraud
% isovalue=4*quantum;
% surf2=isosurface(x,y,z,normalized_Free_Energy_map,isovalue);
% p2 = patch(surf2);
% isonormals(x,y,z,normalized_Free_Energy_map,p2);
% set(p2,'FaceColor','yellow','EdgeColor','none','FaceAlpha',0.3);
% isovalue=2*quantum;
% surf3=isosurface(x,y,z,normalized_Free_Energy_map,isovalue);

%       p3 = patch(surf3);
%       isonormals(x,y,z,normalized_Free_Energy_map,p3);
%       set(p3,'FaceColor','cyan','EdgeColor','none','FaceAlpha',0.2);
%       isovalue=quantum;
%       surf4=isosurface(x,y,z,normalized_Free_Energy_map,isovalue);
%       p4 = patch(surf4);
%       isonormals(x,y,z,normalized_Free_Energy_map,p4);
%       set(p4,'FaceColor','blue','EdgeColor','none','FaceAlpha',0.1);
%             
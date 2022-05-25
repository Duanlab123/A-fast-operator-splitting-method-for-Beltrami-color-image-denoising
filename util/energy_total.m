function energy  = energy_total(beta, eta, u0, v)

v11              = dxf(v(:,:,1)); 
v12              = dyf(v(:,:,1)); 
v21              = dxf(v(:,:,2)); 
v22              = dyf(v(:,:,2)); 
v31              = dxf(v(:,:,3)); 
v32              = dyf(v(:,:,3)); 

z1 = v21.*v32 - v22.*v31;
z2 = v31.*v12 - v11.*v32;
z3 = v11.*v22 - v12.*v21;

temp_mid = sqrt((v11.^2 + v12.^2 + v21.^2 + v22.^2 + v31.^2 + v32.^2) + beta^2.*(z1.^2 + z2.^2 + z3.^2));

norm2 = (u0-v).^2;
energy  = 0.5*eta*sum(norm2(:)) + sum(temp_mid(:));

end
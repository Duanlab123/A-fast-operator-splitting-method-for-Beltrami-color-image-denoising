profile on
clear; close all;

addpath('img'); 
addpath('util');
uclean = double(imread('lena256.png'));
noise                = randn(size(uclean));
noise_std            = 20;  
v0                   = noise_std *noise;    
u0                   = uclean + v0;
u0   = u0/255; uclean = uclean/255;

eta             = 10;           
tau             = 0.02;                        
beta            = 10;  
MaxIter         = 300;   
tol             = 1e-5;


[l1, l2, c]      = size(u0);
v                = u0*1;

P11              = dxf(v(:,:,1)); 
P12              = dyf(v(:,:,1)); 
P21              = dxf(v(:,:,2)); 
P22              = dyf(v(:,:,2)); 
P31              = dxf(v(:,:,3)); 
P32              = dyf(v(:,:,3)); 

y1 = P21.*P32 - P22.*P31;
y2 = P31.*P12 - P11.*P32;
y3 = P11.*P22 - P12.*P21;


relchg         = 1; 
iter           = 1;

tic
t1=clock;
while iter < MaxIter 

 %% ====For P13 ( P^{n+1/3}) ==============
c1 = sqrt((P11.^2 + P12.^2 + P21.^2 + P22.^2 + P31.^2 + P32.^2) + beta^2.*(y1.^2 + y2.^2 + y3.^2));

lambda = max(1 - (tau./c1),0); 

P13_11 = lambda.*P11; 
P13_12 = lambda.*P12;
P13_21 = lambda.*P21; 
P13_22 = lambda.*P22;
P13_31 = lambda.*P31; 
P13_32 = lambda.*P32;

y13_1 = lambda.*y1;
y13_2 = lambda.*y2;
y13_3 = lambda.*y3;


  %% ====For P23 ( P^{n+2/3}) ==============


[P23_11,P23_12,P23_21,P23_22,P23_31,P23_32] = NewtonIter(P13_11,P13_12,P13_21,P13_22,P13_31,P13_32,y13_1,y13_2,y13_3,beta);
 

y23_1 = P23_21.*P23_32 - P23_22.*P23_31;
y23_2 = P23_31.*P23_12 - P23_11.*P23_32;
y23_3 = P23_11.*P23_22 - P23_12.*P23_21;

  %% ====For P33 ( P^{n+3/3}) ==============
z1p              = -1 + exp( sqrt(-1)*2*pi*[0:l1-1]/l1);  %FFT(S1^{+} - I)
z1n              =  1 - exp(-sqrt(-1)*2*pi*[0:l1-1]/l1);  %FFT(-S1^{-} + I)
z2p              = -1 + exp( sqrt(-1)*2*pi*[0:l2-1]/l2);  %FFT(S2^{+} - I)
z2n              =  1 - exp(-sqrt(-1)*2*pi*[0:l2-1]/l2);  %FFT(-S2^{-} + I)

B1    = repmat(conj(z1n').*conj(z1p'), [1 l2]);
B2    = repmat(conj(z2n).*conj(z2p), [l1 1]);

%DtD = abs(psf2otf([1,-1],[l1, l2])).^2 + abs(psf2otf([1;-1],[l1, l2])).^2;

ww        = v;
fft_left  = B1 + B2 - tau*eta; %

fft_right1 = fft2(dxb(P23_11) + dyb(P23_12) - tau*eta*u0(:,:,1));
fft_right2 = fft2(dxb(P23_21) + dyb(P23_22) - tau*eta*u0(:,:,2));
fft_right3 = fft2(dxb(P23_31) + dyb(P23_32) - tau*eta*u0(:,:,3));
v(:,:,1)  = real(ifft2(fft_right1./fft_left));
v(:,:,2)  = real(ifft2(fft_right2./fft_left));
v(:,:,3)  = real(ifft2(fft_right3./fft_left));


v(v < 0)   = 0;
v(v > 1)   = 1;

P33_11    = dxf(v(:,:,1));
P33_12    = dyf(v(:,:,1));
P33_21    = dxf(v(:,:,2));
P33_22    = dyf(v(:,:,2));
P33_31    = dxf(v(:,:,3));
P33_32    = dyf(v(:,:,3));

y33_1 = y23_1;
y33_2 = y23_2;
y33_3 = y23_3;

%% ====Update P and y===================
y1    = y33_1;
y2    = y33_2;
y3    = y33_3;

P11   = P33_11;
P12   = P33_12;
P21   = P33_21;
P22   = P33_22;
P31   = P33_31;
P32   = P33_32;


relchg    = norm(v(:) - ww(:),'fro')/norm(ww(:),'fro'); % relative error

 if relchg < tol
     break;
 end

ReErr(iter) = relchg;
PSNRour(iter) = psnr(v,uclean);
Eng(iter) = energy_total(beta, eta, u0, v); 

t2 = clock;
cput(iter) = etime(t2,t1);

iter = iter + 1;
    
end
t = toc
iternumber = length(ReErr)

% psnr_n = psnr(u0,uclean);
% ssim_n = ssim(u0,uclean);
psnr_v = psnr(v,uclean)
ssim_v = ssim(v,uclean)

figure;imshow(u0,[]);
figure;imshow(v,[]);


figure,
plot(log(ReErr), 'b-','linewidth',2.0); 
xlabel('Iter. No.','FontWeight','bold')
ylabel('ReErr (Log)','FontWeight','bold')
legend('Our')
set(gca,'FontWeight','bold')

figure,
plot(log(Eng), 'g-','linewidth',2.0); 
xlabel('Iter. No.','FontWeight','bold')
ylabel('Energy (Log)','FontWeight','bold')
%legend('Our')
set(gca,'FontWeight','bold')

figure,
plot(PSNRour, 'r-','linewidth',2.0); 
xlabel('Iter. No.','FontWeight','bold')
ylabel('PSNR','FontWeight','bold')
legend('Our')
set(gca,'FontWeight','bold')

figure,
plot(cput(1:length(PSNRour)), PSNRour, 'b-','linewidth',2.0); 
xlabel('CPU time. seconds','FontWeight','bold')
ylabel('PSNR','FontWeight','bold')
legend('Our')
set(gca,'FontWeight','bold')

profile off
profile viewer
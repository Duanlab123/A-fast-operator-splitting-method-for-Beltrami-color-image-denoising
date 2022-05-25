#include "mex.h"
#include<iostream>
#include<cmath>
#include <math.h>
#define N 6
     
using namespace std;
const int N2=2*N;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
    
{
   double *P23_11, *P23_12, *P23_21, *P23_22, *P23_31, *P23_32;
   double *P13_11, *P13_12, *P13_21, *P13_22, *P13_31, *P13_32, *y13_1, *y13_2, *y13_3, beta;
   int m, n, i, j; 
      
   P13_11 = mxGetPr(prhs[0]);
   P13_12 = mxGetPr(prhs[1]);
   P13_21 = mxGetPr(prhs[2]);
   P13_22 = mxGetPr(prhs[3]);
   P13_31 = mxGetPr(prhs[4]);
   P13_32 = mxGetPr(prhs[5]);
   y13_1 = mxGetPr(prhs[6]);
   y13_2 = mxGetPr(prhs[7]);
   y13_3 = mxGetPr(prhs[8]);
   beta = mxGetScalar(prhs[9]);
   
   m = mxGetM(prhs[0]);
   n = mxGetN(prhs[0]);
   
   plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
   P23_11 = mxGetPr(plhs[0]);
   plhs[1] = mxCreateDoubleMatrix(m, n, mxREAL);
   P23_12 = mxGetPr(plhs[1]);
   plhs[2]= mxCreateDoubleMatrix(m, n, mxREAL);
   P23_21 = mxGetPr(plhs[2]);
   plhs[3]=mxCreateDoubleMatrix(m, n, mxREAL);
   P23_22 = mxGetPr(plhs[3]);
   plhs[4] = mxCreateDoubleMatrix(m, n, mxREAL);
   P23_31 = mxGetPr(plhs[4]);
   plhs[5] = mxCreateDoubleMatrix(m, n, mxREAL);
   P23_32 = mxGetPr(plhs[5]);
   
   
    void ff(double yy[N],double *P23_11,double *P23_12,double *P23_21,double *P23_22,double *P23_31,double *P23_32,double *P13_11,double *P13_12,double *P13_21,double *P13_22,double *P13_31,double *P13_32,double *y13_1,double *y13_2,double *y13_3,double beta,int i,int j,int m); 
    void ffjacobian(double yy[N][N],double *P23_11,double *P23_12,double *P23_21,double *P23_22,double *P23_31,double *P23_32,double *y13_1,double *y13_2,double *y13_3,double beta,int i,int j,int m); 
    void inv_jacobian(double yy[N][N],double inv[N][N]); 
    void newdundiedai(double x0[N],double inv[N][N],double y0[N],double x1[N]);
    
    double x0[N],y0[N],jacobian[N][N],invjacobian[N][N],x1[N];

    for(int i = 0; i<m; i++){
         for(int j = 0; j<n; j++){
                 
                 P23_11[j*m+i]=P13_11[j*m+i];
                 P23_12[j*m+i]=P13_12[j*m+i];
                 P23_21[j*m+i]=P13_21[j*m+i];
                 P23_22[j*m+i]=P13_22[j*m+i];
                 P23_31[j*m+i]=P13_31[j*m+i];
                 P23_32[j*m+i]=P13_32[j*m+i];
                 
                 x0[0]=P23_11[j*m+i];
                 x0[1]=P23_12[j*m+i];
                 x0[2]=P23_21[j*m+i];
                 x0[3]=P23_22[j*m+i];
                 x0[4]=P23_31[j*m+i];
                 x0[5]=P23_32[j*m+i];
             
                 ff(y0,P23_11,P23_12,P23_21,P23_22,P23_31,P23_32,P13_11,P13_12,P13_21,P13_22,P13_31,P13_32,y13_1,y13_2,y13_3,beta,i,j,m);

                 ffjacobian(jacobian,P23_11,P23_12,P23_21,P23_22,P23_31,P23_32,y13_1,y13_2,y13_3,beta,i,j,m);
        
                 inv_jacobian(jacobian,invjacobian);

                 newdundiedai(x0,invjacobian,y0,x1);
                 
                 P23_11[j*m+i]=x1[0];
                 P23_12[j*m+i]=x1[1];
                 P23_21[j*m+i]=x1[2];
                 P23_22[j*m+i]=x1[3];
                 P23_31[j*m+i]=x1[4];
                 P23_32[j*m+i]=x1[5];
         }
    }
        
    return;
}           
  
void ff(double yy[N],double *P23_11,double *P23_12,double *P23_21,double *P23_22,double *P23_31,double *P23_32,double *P13_11,double *P13_12,double *P13_21,double *P13_22,double *P13_31,double *P13_32,double *y13_1,double *y13_2,double *y13_3,double beta,int i,int j,int m)
{
        
yy[0] = P23_11[j*m+i]+beta*beta*((P23_22[j*m+i]*P23_22[j*m+i]+P23_32[j*m+i]*P23_32[j*m+i])*P23_11[j*m+i]-(P23_21[j*m+i]*P23_22[j*m+i]+P23_31[j*m+i]*P23_32[j*m+i])*P23_12[j*m+i])+beta*beta*(y13_2[j*m+i]*P23_32[j*m+i]-y13_3[j*m+i]*P23_22[j*m+i])-P13_11[j*m+i]; 
yy[1] = P23_12[j*m+i]+beta*beta*((P23_21[j*m+i]*P23_21[j*m+i]+P23_31[j*m+i]*P23_31[j*m+i])*P23_12[j*m+i]-(P23_21[j*m+i]*P23_22[j*m+i]+P23_31[j*m+i]*P23_32[j*m+i])*P23_11[j*m+i])+beta*beta*(y13_3[j*m+i]*P23_21[j*m+i]-y13_2[j*m+i]*P23_31[j*m+i])-P13_12[j*m+i];
yy[2] = P23_21[j*m+i]+beta*beta*((P23_12[j*m+i]*P23_12[j*m+i]+P23_32[j*m+i]*P23_32[j*m+i])*P23_21[j*m+i]-(P23_11[j*m+i]*P23_12[j*m+i]+P23_31[j*m+i]*P23_32[j*m+i])*P23_22[j*m+i])+beta*beta*(y13_3[j*m+i]*P23_12[j*m+i]-y13_1[j*m+i]*P23_32[j*m+i])-P13_21[j*m+i];
yy[3] = P23_22[j*m+i]+beta*beta*((P23_11[j*m+i]*P23_11[j*m+i]+P23_31[j*m+i]*P23_31[j*m+i])*P23_22[j*m+i]-(P23_11[j*m+i]*P23_12[j*m+i]+P23_31[j*m+i]*P23_32[j*m+i])*P23_21[j*m+i])+beta*beta*(y13_1[j*m+i]*P23_31[j*m+i]-y13_3[j*m+i]*P23_11[j*m+i])-P13_22[j*m+i];
yy[4] = P23_31[j*m+i]+beta*beta*((P23_12[j*m+i]*P23_12[j*m+i]+P23_22[j*m+i]*P23_22[j*m+i])*P23_31[j*m+i]-(P23_11[j*m+i]*P23_12[j*m+i]+P23_21[j*m+i]*P23_22[j*m+i])*P23_32[j*m+i])+beta*beta*(y13_1[j*m+i]*P23_22[j*m+i]-y13_2[j*m+i]*P23_12[j*m+i])-P13_31[j*m+i];
yy[5] = P23_32[j*m+i]+beta*beta*((P23_11[j*m+i]*P23_11[j*m+i]+P23_21[j*m+i]*P23_21[j*m+i])*P23_32[j*m+i]-(P23_11[j*m+i]*P23_12[j*m+i]+P23_21[j*m+i]*P23_22[j*m+i])*P23_31[j*m+i])+beta*beta*(y13_2[j*m+i]*P23_11[j*m+i]-y13_1[j*m+i]*P23_21[j*m+i])-P13_32[j*m+i];    
}   
    
void ffjacobian(double yy[N][N],double *P23_11,double *P23_12,double *P23_21,double *P23_22,double *P23_31,double *P23_32,double *y13_1,double *y13_2,double *y13_3,double beta,int i,int j,int m)
{
    
yy[0][0] = 1+beta*beta*(P23_22[j*m+i]*P23_22[j*m+i]+P23_32[j*m+i]*P23_32[j*m+i]);    
yy[0][1] = -beta*beta*(P23_21[j*m+i]*P23_22[j*m+i]+P23_31[j*m+i]*P23_32[j*m+i]);
yy[0][2] = -beta*beta*P23_22[j*m+i]*P23_12[j*m+i];
yy[0][3] = beta*beta*(2*P23_22[j*m+i]*P23_11[j*m+i]-P23_21[j*m+i]*P23_12[j*m+i]-y13_3[j*m+i]);
yy[0][4] = -beta*beta*P23_32[j*m+i]*P23_12[j*m+i];
yy[0][5] = beta*beta*(2*P23_32[j*m+i]*P23_11[j*m+i]-P23_31[j*m+i]*P23_12[j*m+i]+y13_2[j*m+i]);

yy[1][0] = -beta*beta*(P23_21[j*m+i]*P23_22[j*m+i]+P23_31[j*m+i]*P23_32[j*m+i]);
yy[1][1] = 1+beta*beta*(P23_21[j*m+i]*P23_21[j*m+i]+P23_31[j*m+i]*P23_31[j*m+i]);
yy[1][2] = beta*beta*(2*P23_21[j*m+i]*P23_12[j*m+i]-P23_22[j*m+i]*P23_11[j*m+i]+y13_3[j*m+i]);
yy[1][3] = -beta*beta*P23_21[j*m+i]*P23_11[j*m+i];       
yy[1][4] = beta*beta*(2*P23_31[j*m+i]*P23_12[j*m+i]-P23_32[j*m+i]*P23_11[j*m+i]-y13_2[j*m+i]);     
yy[1][4] = -beta*beta*P23_31[j*m+i]*P23_11[j*m+i];      

yy[2][0] = -beta*beta*P23_12[j*m+i]*P23_22[j*m+i];
yy[2][1] = beta*beta*(2*P23_12[j*m+i]*P23_21[j*m+i]-P23_11[j*m+i]*P23_22[j*m+i]+y13_3[j*m+i]);
yy[2][2] = 1+beta*beta*(P23_12[j*m+i]*P23_12[j*m+i]+P23_32[j*m+i]*P23_32[j*m+i]);
yy[2][3] = -beta*beta*(P23_11[j*m+i]*P23_12[j*m+i]+P23_31[j*m+i]*P23_32[j*m+i]);
yy[2][4] = -beta*beta*P23_32[j*m+i]*P23_22[j*m+i];
yy[2][5] = beta*beta*(2*P23_32[j*m+i]*P23_21[j*m+i]-P23_31[j*m+i]*P23_22[j*m+i]-y13_1[j*m+i]);

yy[3][0] = beta*beta*(2*P23_11[j*m+i]*P23_22[j*m+i]-P23_12[j*m+i]*P23_21[j*m+i]-y13_3[j*m+i]);
yy[3][1] = -beta*beta*P23_11[j*m+i]*P23_21[j*m+i];     
yy[3][2] = -beta*beta*(P23_11[j*m+i]*P23_12[j*m+i]+P23_31[j*m+i]*P23_32[j*m+i]);
yy[3][3] = 1+beta*beta*(P23_11[j*m+i]*P23_11[j*m+i]+P23_31[j*m+i]*P23_31[j*m+i]);
yy[3][4] = beta*beta*(2*P23_31[j*m+i]*P23_22[j*m+i]-P23_32[j*m+i]*P23_21[j*m+i]+y13_1[j*m+i]);   
yy[3][5] = -beta*beta*P23_31[j*m+i]*P23_21[j*m+i];       

yy[4][0] = -beta*beta*P23_12[j*m+i]*P23_32[j*m+i];
yy[4][1] = beta*beta*(2*P23_12[j*m+i]*P23_31[j*m+i]-P23_11[j*m+i]*P23_32[j*m+i]-y13_2[j*m+i]);    
yy[4][2] = -beta*beta*P23_22[j*m+i]*P23_32[j*m+i];
yy[4][3] = beta*beta*(2*P23_22[j*m+i]*P23_31[j*m+i]-P23_21[j*m+i]*P23_32[j*m+i]+y13_1[j*m+i]);      
yy[4][4] = 1+beta*beta*(P23_12[j*m+i]*P23_12[j*m+i]+P23_22[j*m+i]*P23_22[j*m+i]);       
yy[4][5] = -beta*beta*(P23_11[j*m+i]*P23_12[j*m+i]+P23_21[j*m+i]*P23_22[j*m+i]);
 
yy[5][0] = beta*beta*(2*P23_11[j*m+i]*P23_32[j*m+i]-P23_12[j*m+i]*P23_31[j*m+i]+y13_2[j*m+i]);
yy[5][1] = -beta*beta*P23_11[j*m+i]*P23_31[j*m+i];
yy[5][2] = beta*beta*(2*P23_21[j*m+i]*P23_32[j*m+i]-P23_22[j*m+i]*P23_31[j*m+i]-y13_1[j*m+i]);       
yy[5][3] = -beta*beta*P23_21[j*m+i]*P23_31[j*m+i];       
yy[5][4] = -beta*beta*(P23_11[j*m+i]*P23_12[j*m+i]+P23_21[j*m+i]*P23_22[j*m+i]);       
yy[5][5] = 1+beta*beta*(P23_11[j*m+i]*P23_11[j*m+i]+P23_21[j*m+i]*P23_21[j*m+i]);       
}        
  
void inv_jacobian(double yy[N][N],double inv[N][N])
{
    double aug[N][N2],L;
    int i,j,k;

    for (i=0;i<N;i++)
    {  for(j=0;j<N;j++)
            aug[i][j]=yy[i][j];
        for(j=N;j<N2;j++)
            if(j==i+N) aug[i][j]=1;  
            else  aug[i][j]=0;
    }

    for (i=0;i<N;i++)
    {
        for (k=i+1;k<N;k++)
        {L=-aug[k][i]/aug[i][i];
            for(j=i;j<N2;j++)
                aug[k][j]=aug[k][j]+L*aug[i][j];
        }
    }

    for (i=N-1;i>0;i--)
    {
        for (k=i-1;k>=0;k--)
        {L=-aug[k][i]/aug[i][i];
            for(j=N2-1;j>=0;j--)
                aug[k][j]=aug[k][j]+L*aug[i][j];
        }
    }

    for (i=N-1;i>=0;i--)
        for(j=N2-1;j>=0;j--)
            aug[i][j]=aug[i][j]/aug[i][i];

    for (i=0;i<N;i++)
    {  
        for(j=N;j<N2;j++)
            inv[i][j-N]=aug[i][j];
    }

}

void newdundiedai(double x0[N],double inv[N][N],double y0[N],double x1[N])
{
    int i,j;
    double sum=0;
    
    for(i=0;i<N;i++)
    { sum=0;
        for(j=0;j<N;j++)
            sum=sum+inv[i][j]*y0[j];
        x1[i]=x0[i]-sum;
    }
}



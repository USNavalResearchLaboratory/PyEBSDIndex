

float sum16(const float16 *v1);
float sum16(const float16 *v1){
     float sum = 0.0;
     sum += v1[0].s0;
     sum += v1[0].s1;
     sum += v1[0].s2;
     sum += v1[0].s3;
     sum += v1[0].s4;
     sum += v1[0].s5;
     sum += v1[0].s6;
     sum += v1[0].s7;
     sum += v1[0].s8;
     sum += v1[0].s9;
     sum += v1[0].sa;
     sum += v1[0].sb;
     sum += v1[0].sc;
     sum += v1[0].sd;
     sum += v1[0].se;
     sum += v1[0].sf;
     return sum;

}


void print16(const float16 v1);
void print16(const float16 v1){
     
     printf( "%f, ", v1.s0);
     printf( "%f, ", v1.s1);
     printf( "%f, ", v1.s2);
     printf( "%f, ", v1.s3);
     printf( "%f, ", v1.s4);
     printf( "%f, ", v1.s5);
     printf( "%f, ", v1.s6);
     printf( "%f, ", v1.s7);
     printf( "%f, ", v1.s8);
     printf( "%f, ", v1.s9);
     printf( "%f, ", v1.sa);
     printf( "%f, ", v1.sb);
     printf( "%f, ", v1.sc);
     printf( "%f, ", v1.sd);
     printf( "%f, ", v1.se);
     printf( "%f", v1.sf);

}


__kernel void nlloadpat8bit(__global uchar *datain, __global float *dataout){
  const unsigned long int x = get_global_id(0);
  uchar imVal =  datain[x];
  float imValflt = convert_float(imVal);
  dataout[x] = imValflt;
}

__kernel void nlloadpat16bit(__global ushort *datain, __global float *dataout){
  const unsigned long int x = get_global_id(0);
  ushort imVal = datain[x];
  float imValflt = convert_float(imVal);
  dataout[x] = imValflt;
}

__kernel void nlloadpat32flt(__global float *datain, __global float *dataout){
  const unsigned long int x = get_global_id(0);
  dataout[x] = datain[x];
}

__kernel void calcsigma( __global float *data, __global float16 *mask, 
      __global float *sig, 
      __global float *dg, __global float *ng,
      const long  nn, const long ndatchunk, const long npatpoint, const float maxlim){
  // IDs of work-item represent x and y coordinates in image
  const long x = get_global_id(0);
  const long y = get_global_id(1);
  const long ncol = get_global_size(0);
  const long nrow = get_global_size(1);
  const long indx_xy = x+y*ncol;
  
  const long indx0 = npatpoint * indx_xy;

  long i, j, z;
  long indx_j, indx_ij, count; 

  long nnn = (2*nn+1) * (2*nn+1);

  float16 d1, d0; 
  float16 maskchunk; 
  float16 mask0, mask1;
  float dd; 


  float d[256];
  float n[256]; 
 
  
  for(j=0; j < nnn; ++j){
      d[j] = 0.0;
      n[j] = 1.0e-6; 
  }

  ;

  for(z = 0; z<ndatchunk; ++z){
      count = 0; 
      
      
      d0 =  *(__global float16*) (data + indx0+16*z); 
      //if (z == 0){print16(d0);}
      //
      maskchunk = mask[z];
      //print16(maskchunk);
      mask0 =  d0 < (float16) (maxlim) ? (float16) (1.0): (float16) (0.0);
      mask0 =  maskchunk > (float16) 1.e-3 ? mask0 : (float16) 0.0; 
      //mask0 = select((float16) (1.0), (float16) (0.0), isless( d0, (float16) maxlim));
      
      for(j=y-nn; j<=y+nn; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-nn; i<=x+nn; ++i){      
            
              indx_ij =  (i >= 0) ? (i): abs(i);
              indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
              indx_ij = npatpoint*(indx_ij + indx_j);
              
              mask1 = mask0;
              d1 =  *(__global float16*) (data + indx_ij+16*z); 
              

              mask1 =  d1 < (float16) (maxlim) ? mask1 : (float16) (0.0);
              //mask1 = select((float16) (1.0), (float16) (0.0), isgreater(mask0, (float16)(1e-6)) && isgreater(mask1,(float16)(1e-6)));
              //printf("%*s\n", 'here');
              
              d1 = (d0-d1);
              d1 *= d1 ; 
              d1 *= mask1; 

              dd = sum16(&d1);
              dd = (indx_ij == indx0) ? -1*dd : dd; // mark the center point 
              
              d[count] += dd;
              n[count] += sum16(&mask1);
              //d[count+nnn*indx_xy] += dd;
              //n[count+nnn*indx_xy] += sum16(&mask1);

              count += 1; 
          }
      }
      //printf("%d %f\n", count, n[count]);
  }


  float mind = 1e24; 
  float s0;  
  ; //nnn*(x+ncol*y);
  for(j=0; j<nnn; ++j){
    if (d[j] > 1e-5){ //sometimes EDAX collects the same pattern twice & catch the pixel of interest.
      s0 = d[j]/(2.0*n[j]); 
      if (s0<mind){
        mind = s0; 
      }
    } 
  }

  sig[indx_xy] = sqrt(mind);
  
  for(j=0; j<nnn; ++j){
    dg[j+nnn*indx_xy] = d[j];
    ng[j+nnn*indx_xy] = n[j];
  }

}


__kernel void normd(
    const __global float *sigma, 
    const __global float *n, 
     __global float *d,
     const long sr){

    const long x = get_global_id(0);
    const long y = get_global_id(1);
    const long ncol = get_global_size(0);
    const long nrow = get_global_size(1);
    const long indx_xy = x+y*ncol;

    long i, j, q;
    long indx_j, indx_ij; 

    long nnn = (2*sr+1) * (2*sr+1);

    float sigma_xy = sigma[indx_xy];  
    sigma_xy *= sigma_xy;
    //printf("%f", sigma_xy);
    float sigma_ij, nn, dd;  

    for(j=y-nn; j<=y+nn; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-nn; i<=x+nn; ++i){  
              
               indx_ij =  (i >= 0) ? (i): abs(i);
               indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
               indx_ij =  (indx_ij + indx_j);
               sigma_ij = sigma[indx_ij];
               sigma_ij *= sigma_ij;
              
               sigma_ij = sigma_ij + sigma_xy;
               for(q=0;q<nnn;q++){
                dd = d[q+nnn*indx_xy];
                nn = n[q+nnn*indx_xy];    
                if (nn > 1.0e-3){           
                  dd -= nn*sigma_ij;
                  dd /= (sigma_ij * sqrt(2.0*nn)); 
                  //printf("%f\n", dd) ;
                  d[q+nnn*indx_xy] = dd;    
               }
                    
             }
             
           }

     }      


}




__kernel void calcnlpar( 
      const __global float *data, 
      const __global float16 *mask, 
      const __global float *sigma, 
      const __global long *crlimits, 
      __global float *dataout, 
      const long sr, 
      const long ndatchunk, 
      const long npatpoint, 
      const float maxlim, 
      const float lam2, 
      const float dthresh){
  //IDs of work-item represent x and y coordinates in image
  //const long4 calclim =  crlimits[0];
  const long x = get_global_id(0)+crlimits[0];
  const long y = get_global_id(1)+crlimits[1];
  const long ncol = crlimits[2]; //get_global_size(0);
  const long nrow = crlimits[3];//get_global_size(1);
  const long indx_xy = x+y*ncol;
  //printf("%d\n", indx_xy);
  const long indx0 = npatpoint * indx_xy;
  //printf("%d, %d, %d, %d\n", x,y,ncol, nrow); 
  long i, j, z;
  long indx_j, indx_ij, count; 

  long nnn = (2*sr+1) * (2*sr+1);

  float16 d1, d0; 
  float16 mask0, mask1;
  float dd, nn, sigma_ij, norm; 
  float sigma0 = sigma[indx_xy];
  sigma0 *= sigma0; 
  //float16* writeloc; 

  float d[512]; // taking a risk here that noone will want a SR > 10
  float n[512]; 

  
  for(j=0; j < nnn; ++j){
      d[j] = 0.0;
      n[j] = 1.0e-6; 
  }


  for(z = 0; z<ndatchunk; ++z){
      count = 0; 
      
      
      d0 =  *(__global float16*) (data + indx0+16*z); 
      
      mask0 = mask[z];
      mask0 =  (mask0 > (float16) 1.e-3) ? (float16) 1.0 : (float16) 0.0; 
      mask0 =  d0 < (float16) (maxlim) ? mask0 : (float16) (0.0);
      
      
      for(j=y-sr; j<=y+sr; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-sr; i<=x+sr; ++i){
            
              indx_ij =  (i >= 0) ? (i): abs(i);
              indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
              
              indx_ij = (indx_ij + indx_j);
              
              indx_ij *= npatpoint;
              
              mask1 = mask0;
              d1 =  *(__global float16*) (data + indx_ij+16*z); 
              

              mask1 =  d1 < (float16) (maxlim) ? mask1 : (float16) (0.0);
              //mask1 = select((float16) (1.0), (float16) (0.0), isgreater(mask0, (float16)(1e-6)) && isgreater(mask1,(float16)(1e-6)));
              //printf("%*s\n", 'here');
              
              d1 = (d0-d1);
              d1 *= d1 ; 
              d1 *= mask1; 

              dd = sum16(&d1);
              dd = (indx_ij == indx0) ? -1.0 : dd; // mark the center point 
              
              d[count] += dd;
              n[count] += sum16(&mask1);
              //d[count+nnn*indx_xy] += dd;
              //n[count+nnn*indx_xy] += sum16(&mask1);

              count += 1; 
          }
      }
      
  }
// calculate the weights
  count = 0; 
  float sum = 0.0; 
  nn = 1.0;
  for(j=y-sr; j<=y+sr; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-sr; i<=x+sr; ++i){
              
              dd = d[count]; 
              if (dd > 1.e-3){
                indx_ij =  (i >= 0) ? (i): abs(i);
                indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
                
                indx_ij = (indx_ij + indx_j);
                
                sigma_ij = sigma[indx_xy];
                sigma_ij *= sigma_ij; 
                nn = n[count]; 
                dd -= nn*(sigma_ij+sigma0);
                
                norm = (sigma0 + sigma_ij)*sqrt(2.0*nn);
                if (norm > 1.0e-8){
                  dd /= norm;  
                } else {
                  dd = 1e6*nn;
                } 

              } else {
                dd = 0.0;
                nn = 1.0;
              }
              
              dd -= dthresh; 
              dd = dd >= 0.0 ? dd : 0.0; 

              dd = exp(-1.0*dd*lam2); 
              sum += dd; 
              d[count] = dd;

              count += 1; 
              
          }
  }
  
  for (j =0;j<nnn;j++){d[j] *= 1.0/sum; }

// now one more time through the loops to average across the patterns with the weights  
// and place in the output array
  for(z = 0; z<ndatchunk; ++z){
      count = 0; 
      d0 =  *(__global float16*) (data + indx0+16*z); 
      
      mask0 = mask[z];
      mask0 =  (mask0 > (float16) -1.e-3) ? (float16) 1.0 : (float16) 0.0; 
      
      for(j=y-sr; j<=y+sr; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-sr; i<=x+sr; ++i){
            
              indx_ij =  (i >= 0) ? (i): abs(i);
              indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
              
              indx_ij = (indx_ij + indx_j);
              
              indx_ij *= npatpoint;
              
              
              d1 =  *(__global float16*) (data + indx_ij+16*z); 
              
              
              d1 *= (float16) d[count]; 
              d1 *= mask0; 
              
              *(__global float16*) (dataout + indx0+16*z) += d1; 
              
              //writeadd16(&(dataout[indx0+16*z]), &d1 );

              //d[count+nnn*indx_xy] += dd;
              //n[count+nnn*indx_xy] += sum16(&mask1);

              count += 1; 
          }
      }
  }



}






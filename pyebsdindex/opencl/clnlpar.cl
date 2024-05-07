

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


__kernel void nlloadpat8bit(const __global uchar *datain, __global float *dataout){
  const unsigned long int x = get_global_id(0);
  uchar imVal;
  float imValflt;
  imVal =  datain[x];
  imValflt = convert_float(imVal);
  dataout[x] = imValflt;
}

__kernel void nlloadpat16bit(const __global ushort *datain, __global float *dataout){
  const unsigned long int x = get_global_id(0);
  ushort imVal;
  float imValflt;
  imVal =  datain[x];
  imValflt = convert_float(imVal);
  dataout[x] = imValflt;
}

__kernel void nlloadpat32flt(const __global float *datain, __global float *dataout){
  const unsigned long int x = get_global_id(0);
  dataout[x] = datain[x];
}

__kernel void calcsigma( const __global float *data, const __global float16 *mask, 
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




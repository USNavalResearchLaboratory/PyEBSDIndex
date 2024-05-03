float sum16(const float16 v1);
float sum16(const float16 v1){
     float sum = 0.0;
     sum += v1.s0;
     sum += v1.s1;
     sum += v1.s2;
     sum += v1.s3;
     sum += v1.s4;
     sum += v1.s5;
     sum += v1.s6;
     sum += v1.s7;
     sum += v1.s8;
     sum += v1.s9;
     sum += v1.sa;
     sum += v1.sb;
     sum += v1.sc;
     sum += v1.sd;
     sum += v1.se;
     sum += v1.sf;
     return sum;

}


__kernel void calcsigma( __global const float16 *data, __global float *sigma, const long  nn, \
      const long ndatchunk, const float maxlim){
  // IDs of work-item represent x and y coordinates in image
  const long x = get_global_id(0);
  const long y = get_global_id(1);
  const long int ncol = get_global_size(0);
  const long int nrow = get_global_size(1);
  //const unsigned long int ndat = ndatchunk*16; 
  
  long i, j, z, indx_y, indx_x;
  unsigned long indxyz, indx0, count; 
  //unsigned long nnn = (2*nn+1) * (2*nn+1); 
  
  float16 d1, d0; 
  float16 mask0, mask1;
  


  __local float d[128];
  __local float n[128]; 
  count = 0; 
  for(j=0; j<nn; ++j){
    for (i=0; i<nn; ++i){
      d[count] = 0.0;
      n[count] = 0.0; 
    }
  }



  for(z = 0; z<ndatchunk; z++){
      count = 0; 
      indx0 = z + ndatchunk * (x + ncol * y);
      d0 =  data[indx0]; 
      
      
      mask0 = select((float16) (1.0), (float16) (0.0), isless( d0, (float16) (maxlim)));
      for(j=y-nn; j<j+nn; ++j){
          indx_y =  (j > 0) ? (j): -1*j;
          indx_y =  (indx_y < nrow) ? (indx_y): nrow - (indx_y -nrow +1);
          indx_y = ncol*indx_y;
          
          for(i=x-nn; i<x+nn; ++i){      
            
              indx_x =  (i > 0) ? (i): -1*i;
              indx_x =  (indx_x < ncol) ? (indx_x): ncol - (indx_x -ncol +1);
              indx_x = ndatchunk*(indx_x + indx_y);
              
              
              indxyz = z + indx_x;
              d1 = data[indxyz];
              
              mask1 = select((float16) (1.0), (float16) (0.0), isless( d1, (float16)maxlim));
              mask1 = select((float16) (1.0), (float16) (0.0), isgreater(mask0, (float16)(1e-6)) && isgreater(mask1,(float16)(1e-6)));
              d1 = (d0-d1);
              d1 = d1 * d1; 
              
              d[count] += sum16(d1);
              n[count] += sum16(mask1);
              
              
              count += 1; 
          }
      }
  }
      
}
  
  


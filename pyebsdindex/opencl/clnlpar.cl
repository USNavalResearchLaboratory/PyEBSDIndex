
__kernel float dot16(float16 *v1, float16 *v2){
  float8 v1_8 = *v1.low;
  float8 v2_8 = *v2.low;

  float d16 = 0.0;
  
  d16 += dot((float4) v1_8.low, (float4) v2_8.low) +\
       dot((float4) v1_8.high, (float4) v2_8.high)

  v1_8 = *v1.high;
  v2_8 = *v2.high;
  
  d16 += dot((float4) v1_8.low, (float4) v2_8.low) +\
       dot((float4) v1_8.high, (float4) v2_8.high)

   return d16
}




__kernel void calcsigma( __global const float16 *data, __global float *sigma, const long  nn, \
      const long ndatchunk, const float maxlim)
{
  // IDs of work-item represent x and y coordinates in image
  const long x = get_global_id(0);
  const long y = get_global_id(1);
  const long int ncol = get_global_size(0);
  const long int nrow = get_global_size(1);
  //const unsigned long int ndat = ndatchunk*16; 
  
  long i, j, z, indx_y, indx_x, q;
  unsigned long indxyz, indx0, count; 
  unsigned long nnn = (2*nn+1) * (2*nn+1); 
  
  float16 d1, d0, mask0, mask1;
  float4 ones = (float16)1.0;
  float4 temp4; 
  
  float d[128] = {0.0};
  float temp; 
  //float sigtemp[(2*nn+1) * (2*nn+1)] = {1e18}; 
  float n[128] = {0.0}; 

  for(z = 0; z<ndatchunk; z++)
    {
      count = 0; 
      indx0 = z + ndatchunk * (x + ncol * y);
      d0 =  data[indx0]; 
      
      //mask0 = (condition) ? 1.0: 0.0;
      mask0 = select((float16) 1.0, (float16) 0.0, isless( d0, (float16)maxlim));
      for(j=y-nn; j<j+nn; ++j)
      {
          indx_y =  (j > 0) ? (j): -1*j;
          indx_y =  (indx_y < nrow) ? (indx_y): nrow - (indx_y -nrow +1);
          indx_y = ncol*indx_y;
          
          for(i=x-nn; i<x+nn; ++i)
          {      
            
              indx_x =  (i > 0) ? (i): -1*i;
              indx_x =  (indx_x < ncol) ? (indx_x): ncol - (indx_x -ncol +1);
              indx_x = ndatchunk*(indx_x + indx_y);
              
              
              indxyz = z + indx_x;

              d1 = data[indxyz];
              
              mask1 = select((float16) 1.0, (float16) 0.0, isless( d1, (float16)maxlim));
              mask1 = select((float16) 1.0, (float16) 0.0, isgreater(mask0, (float16)1e-6) && isgreater(mask1,(float16)1e-6));
              d1 = (d0-d1);
              d1 = d1 * d1; 
              temp = dot16(d1,ones);
              d[count] += temp;
              temp = dot16(mask1,ones);
              n[count] += temp;
              count += 1; 
          }
        }
      }
      
}
  
  


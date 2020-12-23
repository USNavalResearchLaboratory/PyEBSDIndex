


 __kernel void radonSum(
      __global const unsigned long int *rdnIndx, __global const float *images, __global float *radon, 
      const unsigned long int imstep, const unsigned long int indxstep, const unsigned long int rdnstep )
  {
    unsigned long int gid_im = get_global_id(0);
    unsigned long int gid_rdn = get_global_id(1);
    unsigned long int i, k, j, idx;
    float sum, count;
    
    k = gid_rdn+gid_im*rdnstep;
    sum = 0.0;
    count = 0.0;
    j = gid_im*imstep;
    for (i=0; i<indxstep; i++){  
      idx = rdnIndx[gid_rdn*indxstep+i];
      if (idx >= imstep) {
        break;
      } else {
        sum += images[j + idx];
        count += 1.0;
      } 
    }      
    radon[k] = sum/(count + 1.0e-12);
  }

__kernel void morphDilateKernel( __global const float *in, const int imszx, const int imszy, const int imszz, 
  const int kszx, const int kszy, __global float *out)
{
  //IDs of work-item represent x and y coordinates in image; z is the n-th image.
  const long int x = get_global_id(0);
  const long int y = get_global_id(1);
  const long int z = get_global_id(2);
  long int yIndx; 
  float pxVal;
  const int kszx2 = kszx/2;
  const int kszy2 = kszy/2;

  const long int istart = x-kszx2 >= 0 ? x-kszx2: 0;
  const long int iend = x+kszx2 < imszx ? x+kszx2: imszx-1;
  const long int jstart = y-kszy2 >= 0 ? y-kszy2: 0;
  const long int jend = y+kszy2 < imszy ? y+kszy2: imszy-1;

  // extreme pixel value (max/min for dilation/erosion) inside structuring element
  float extremePxVal = -1.0e12; // initialize extreme value with min when searching for max value
  
  const long int zIndx = imszx*imszy*z;
  
  for(int j=jstart; j<=jend; ++j)
  {
      yIndx = zIndx + (j)*imszx;
      
      for(int i=istart; i<=iend; ++i)
      {      
      pxVal = in[yIndx+i];
      //const float pxVal = read_imagef(in, sampler_dilate, (int3)(x + i, y + j, z)).s0;
      extremePxVal = max(extremePxVal, pxVal);
    }
  }
  
  // write value of pixel to output image at location (x, y,z)
  out[zIndx + y*imszx + x] = extremePxVal;
}




__kernel void convolution3d2d( __global const float *in, __constant float *kern, const int imszx, const int imszy, const int imszz, 
  const int kszx, const int kszy, const int padx, const int pady, __global float *out)
{
  // IDs of work-item represent x and y coordinates in image
  const int x = get_global_id(0)+padx;
  const int y = get_global_id(1)+pady;
  const long int z = get_global_id(2);
  //long int indxIm; 
  //long int indxK;
  long int yIndx; 
  long int yKIndx; 
  float pxVal, kVal;
  const int kszx2 = kszx/2;
  const int kszy2 = kszy/2;

  //const long int istart = ((x-kszx2) >= 0) ? x-kszx2: 0;
  //const long int iend = ((x+kszx2) < imszx) ? x+kszx2: imszx-1;
  //const long int jstart = ((y-kszy2) >= 0) ? y-kszy2: 0;
  //const long int jend = ((y+kszy2) < imszy) ? y+kszy2: imszy-1;
  const long int istart = ((x+kszx2) >= imszx) ? x+kszx2+1-imszx: 0;
  const long int iend = ((x-kszx2) < 0) ? x+kszx2+1: kszx;
  const long int startx = x+kszx2;
  const long int jstart = ((y+kszy2) >= imszy) ? y+kszy2+1-imszy: 0;
  const long int jend = ((y-kszy2) < 0) ? y+kszy2+1: kszy;
  const long int starty = y+kszy2;
  
  float sum = 0.0f; // initialize convolution sum
  
  const long int zIndx = imszx*imszy*z;
  float current = out[zIndx + y*imszx + x];

  for(int j=jstart; j<jend; ++j)
  {
      //yIndx = zIndx + (j)*imszx;
      //yKIndx = (j-y)*kszx;
      yKIndx = kszx*(j);
      yIndx = zIndx + imszx*(starty - j);
      for(int i=istart; i<iend; ++i)
      {      
        pxVal = in[yIndx+(startx - i)];
        kVal = kern[yKIndx + i];
        sum += pxVal*kVal;
    }
  }
  sum = current < sum ? sum : current;
  // write value of pixel to output image at location (x, y,z)
  out[zIndx + y*imszx + x] = sum;
}

__kernel void im1NEim2( __global const float *im1, __global const float *im2, __global uchar *out, const int imszx, const int imszy, const int imszz)
{
  // IDs of work-item represent x and y coordinates in image
  const long int x = get_global_id(0);
  const long int y = get_global_id(1);
  const long int z = get_global_id(2);
  

  const long int indx = x + imszx*y + imszx*imszy*z ;
  const float im1val = im1[indx];
  const float im2val = im2[indx];
  const float diff = fabs(im1val - im2val);
  //diff = (im2val < 1.0e-6)? 10.0: diff; // must be larger than 0
  //const float scale = im1val < im2val ? im1val:im2val;
  //out[indx] = ((diff/scale) < 1e-8)? 1.0 : 0.0;
  out[indx] = (diff < 1e-6) ? 1 : 0;
  
  
}


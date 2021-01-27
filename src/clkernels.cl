


 __kernel void radonSum2( // turns out this is significantly slower.
 // something about allocating the additional set of loops?
      __global const unsigned long int *rdnIndx, __global const float *images, __global float *radon, 
      const unsigned int imstep, const unsigned int indxstep,
      const unsigned int nRhoP, const unsigned int nThetaP,
      const unsigned int rhoPad, const unsigned int thetaPad)
  {
    float sum = 0.0, count = 0.0;
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int gid_rho = get_global_id(1);
    const unsigned long int gid_theta = get_global_id(2);
    const unsigned long int nTheta = get_global_size(2);

    const unsigned long int imStart = imstep * gid_im;
    //const unsigned long int gid_rdn = gid_theta + nTheta * gid_rho;
    //const unsigned long int indxStart = gid_rdn*indxstep;
    const unsigned long int indxStart = (gid_theta + nTheta * gid_rho)*indxstep;
    unsigned long int i, idx;

    for (i=0; i<indxstep; i++){  
      idx = rdnIndx[indxStart+i];
      if (idx >= imstep) {
        break;
      } else {
        sum += images[imStart + idx];
        count += 1.0;
      } 
    }
  }

__kernel void radonSum(
      __global const unsigned long int *rdnIndx, __global const float *images, __global float *radon,
      const unsigned int imstep, const unsigned int indxstep,
      const unsigned int nRhoP, const unsigned int nThetaP,
      const unsigned int rhoPad, const unsigned int thetaPad,const unsigned int nTheta )
    {
    float sum = 0.0, count = 0.0;
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int q = get_global_id(1);
    const unsigned long int gid_rho = q/nTheta;
    const unsigned long int gid_theta = q % nTheta;

    const unsigned long int imStart = imstep * gid_im;
    const unsigned long int indxStart = (gid_theta + nTheta * gid_rho)*indxstep;
    unsigned long int i, idx;

    for (i=0; i<indxstep; i++){
      idx = rdnIndx[indxStart+i];
      if (idx >= imstep) {
        break;
      } else {
        sum += images[imStart + idx];
        count += 1.0;
      }
    }

    const unsigned long int gid_rdnP = (nThetaP * nRhoP * gid_im) + (nThetaP * (gid_rho+rhoPad)) + (gid_theta + thetaPad) ;
    radon[gid_rdnP] = sum/(count + 1.0e-12);
    }

  __kernel void radonFixArt(
      __global float *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int thetaPad)
  {
    unsigned long int gid_im = get_global_id(0);
    unsigned long int gid_rho = get_global_id(1);
    unsigned long int gid_rdn;
    gid_rdn = nTheta * nRho * gid_im + (nTheta * gid_rho);

    radon[gid_rdn+thetaPad] = radon[gid_rdn+thetaPad+1];
    radon[gid_rdn+nTheta-1-thetaPad] = radon[gid_rdn+nTheta-2-thetaPad];
  }

// Padding of the radon Theta -- 0 and 180* are symmetric with a vertical flip.
__kernel void radonPadTheta(
      __global float *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int thetaPad)
  {
    unsigned long int gid_im = get_global_id(0);
    unsigned long int gid_rho = get_global_id(1);
    unsigned long int gid_rdn1,gid_rdn2, i, indxim;
    indxim = nTheta * nRho * gid_im;
    gid_rdn1 =  indxim + (nTheta * gid_rho);
    gid_rdn2 =  indxim + (nTheta * (nRho -1 - gid_rho));

    for (i = 0; i <= thetaPad; ++i){
        radon[gid_rdn1 + i] = radon[gid_rdn2 + (nTheta-1 - (2 * thetaPad)) + i];
        radon[gid_rdn2 + (nTheta-1 - ( thetaPad)) + i] = radon[gid_rdn1 + thetaPad + i];
    }

  }

// Padding of the radon Rho -- copy the defined line to all other rows ...
  __kernel void radonPadRho(
      __global float *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int rhoPad)
  {
    unsigned long int gid_im = get_global_id(0);
    unsigned long int gid_theta = get_global_id(1);
    unsigned long int indxim, i, gid_rdn1, gid_rdn2;
    float rd1p, rd2p;
    indxim = nTheta * nRho * gid_im;
    rd1p =  radon[indxim + (nTheta * rhoPad) + gid_theta] ;
    rd2p =  radon[ indxim + (nTheta * (nRho -1 - rhoPad)) + gid_theta] ;

    for (i = 0; i <= rhoPad; ++i){

        gid_rdn1 = indxim + (nTheta*i) + gid_theta;
        gid_rdn2 = indxim + (nTheta* (nRho-1-rhoPad+i)) + gid_theta;
        radon[gid_rdn1] = rd1p;
        radon[gid_rdn2] = rd2p;
    }

  }

// Convolution of a stack of images by a 2D kernel
// At somepoint we might want to consider the ability to chain together convolutions -- keeping the max at each pixel...
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

  const long int istart = ((x+kszx2) >= imszx) ? x+kszx2+1-imszx: 0;
  const long int iend = ((x-kszx2) < 0) ? x+kszx2+1: kszx;
  const long int startx = x+kszx2;
  const long int jstart = ((y+kszy2) >= imszy) ? y+kszy2+1-imszy: 0;
  const long int jend = ((y-kszy2) < 0) ? y+kszy2+1: kszy;
  const long int starty = y+kszy2;
  
  float sum = 0.0f; // initialize convolution sum
  
  const long int zIndx = imszx*imszy*z;
  //float current = out[zIndx + y*imszx + x];

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
  // sum = current < sum ? sum : current;
  // write value of pixel to output image at location (x, y,z)
  out[zIndx + y*imszx + x] = sum;
}

//Brute force gray-scale dilation (or local max).  This will be replaced with the van Herk/Gil-Werman algorithm
__kernel void morphDilateKernelBF( __global const float *in,  __global float *out,
                                   const uint imszx, const uint imszy, const uint xpad, const uint ypad,
                                   const uint kszx, const uint kszy)
{
  //IDs of work-item represent x and y coordinates in image; z is the n-th image.
  const long int x = get_global_id(0) + xpad;
  const long int y = get_global_id(1) + ypad;
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

//find the minimum value in each image.  This probably could be sped up by using a work group store.
__kernel void imageMin( __global const float *im1, __global float *imMin,
                        const int imszx, const int imszy, const int padx, const int pady)
  {
  // IDs of work-item represent x and y coordinates in image
  const long int z = get_global_id(0);
  long int indx,i, j, indxz;
  float cmin = 1.0e12;
  float imVal;

  indxz = z*imszx*imszy;
  for(j = pady; j<= imszy - pady-1; ++j){
    indx = indxz + j*imszx;
    for(i = padx; i<= imszx - padx-1; ++i){
        imVal = im1[indx+i];
        if (imVal < cmin){
            cmin = imVal;
        }
    }
  }
  imMin[z] = cmin;
}

// Subtract a value from an image stack, with clipping.
// The value to be subtracted are unique to each image, stored in an array in imMin
__kernel void imageSubMinWClip( __global float *im1, __global const float *imMin,
  const int imszx, const int imszy, const int padx, const int pady)
  {
  // IDs of work-item represent x and y coordinates in image
  const long int x = get_global_id(0) + padx;
  const long int y = get_global_id(1) + pady;
  const long int z = get_global_id(2);

  const long int indx = x + imszx*y + imszx*imszy*z ;
  const float im1val = im1[indx];
  float value;
  value = im1val - imMin[z];
  im1[indx] = value < 0.0 ? 0.0 : value;

}



// Is image1 (can be a stack of images) EQ to image2 (can be a stack) -- returns byte array.
// There is ability to include padding in x and y
__kernel void im1EQim2( __global const float *im1, __global const float *im2, __global uchar *out,
                        const uint imszx, const uint imszy, const uint padx, const uint pady)
{
  // IDs of work-item represent x and y coordinates in image
  const long int x = get_global_id(0)+padx;
  const long int y = get_global_id(1)+pady;
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

void morphDilateKernel(const long winsz, const long winsz2,
                      __local float *s,  __local float *r,  __local float *w);

void morphDilateKernel(const long winsz, const long winsz2,
                      __local float *s,  __local float *r,  __local float *w){
  long int i;
  const int winsz1 = winsz-1;

  s[winsz1] = w[winsz1];
  r[0] = w[winsz1];
  for (i = 1; i< winsz; i++){
    s[winsz1-i] = max(s[winsz1-(i-1)], w[winsz1-i] );
    r[i] = max(r[i-1], w[winsz1+i]);
  }

  for (i = 0; i< winsz; i++){
        w[i+winsz2] = max(s[i], r[i]);
  }
}

__kernel void morphDilateKernelX( __global const float *im1, __global float *out,
                        const long winszX, const long winszX2,
                        const unsigned int imszx, const unsigned int imszy,
                        const unsigned int padx, const unsigned int pady,
                        __local float *s, __local float *r, __local float *w)
{
  // IDs of work-item represent x and y coordinates in image
  const long int chunknum = get_global_id(0);
  const long int y = get_global_id(1)+pady;
  const long int z = get_global_id(2)*imszx*imszy;
  //float s[128];
  //float r[128];
  //float w[256];

  long int i;
  const long int indx = (padx + chunknum*winszX - winszX2)  + imszx*y + z ;

  for (i=0; i<winszX; i++){
    s[i] = -1.0e12;
    r[i] = -1.0e12;
  }

  for (i =0; i<(winszX*2-1); ++i){
    w[i] = im1[indx+i];
  }

  morphDilateKernel(winszX, winszX2,s, r,w);
  for (i =winszX2; i<(winszX2+winszX); ++i){
    out[indx+i] = w[i];
  }

}


__kernel void morphDilateKernelY( __global const float *im1, __global float *out,
                        const long winszY, const long winszY2,
                        const unsigned int imszx, const unsigned int imszy,
                        const unsigned int padx, const unsigned int pady,
                        __local float *s, __local float *r, __local float *w)
{
  // IDs of work-item represent x and y coordinates in image
  const long int x = get_global_id(0) + padx;
  const long int chunknum = get_global_id(1);
  const long int z = get_global_id(2)*imszx*imszy;

  //float s[128];
  //float r[128];
  //float w[256];


  long int i;
  const long int indx = x + (pady + chunknum*winszY - winszY2) * imszx + z;

  for (i=0; i<winszY; i++){
    s[i] = -1.0e12;
    r[i] = -1.0e12;
  }

  for (i =0; i<(winszY*2-1); ++i){
    w[i] = im1[indx+(i*imszx)];
  }

  morphDilateKernel(winszY, winszY2,s, r,w);
  for (i =winszY2; i<(winszY2+winszY); ++i){
    out[indx+(i*imszx)] = w[i];
  }

}



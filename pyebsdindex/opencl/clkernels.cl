/*
This software was developed by employees of the US Naval Research Laboratory (NRL), an
agency of the Federal Government. Pursuant to title 17 section 105 of the United States
Code, works of NRL employees are not subject to copyright protection, and this software
is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
responsibility whatsoever for its use by other parties, and makes no guarantees,
expressed or implied, about its quality, reliability, or any other characteristic. We
would appreciate acknowledgment if the software is used. To the extent that NRL may hold
copyright in countries other than the United States, you are hereby granted the
non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
works and distribute this software, in any medium, or authorize others to do so on your
behalf, on a royalty-free basis throughout the world. You may improve, modify, and
create derivative works of the software or any portion of the software, and you may copy
and distribute such modifications or works. Modified works should carry a notice stating
that you changed the software and should note the date and nature of any such change.
Please explicitly acknowledge the US Naval Research Laboratory as the original source.
This software can be redistributed and/or modified freely provided that any derivative
works bear some notice that they are derived from it, and any modified versions bear
some notice that they have been modified.

Author: David Rowenhorst; 
The US Naval Research Laboratory Date: 21 Aug 2020
*/


//Do a background subtract on the pattern
__kernel void backSub( __global float16 *im1, __global const float *back,
                        const unsigned long int nImChunk)
  {
  const unsigned long int xy = get_global_id(0);
  //const unsigned long int szim = get_global_size(0);
  unsigned long i;
  float16 imVal;

  const float b1 = back[xy];

  const unsigned long indx = nImChunk * xy;
  for(i = 0; i< nImChunk; ++i){
    imVal =  im1[indx+i];
    imVal -= b1;   
    im1[indx+i] = imVal;
  }
  
}


__kernel void radonSum(
      __global const unsigned long int *rdnIndx, __global const float16 *images, __global float16 *radon,
      const unsigned long int imstep, const unsigned long int indxstep,
      const unsigned int long nRhoP, const unsigned long int nThetaP,
      const unsigned long int rhoPad, const unsigned long int thetaPad,const unsigned long int nTheta )
    {

    float16 sum = (float16)(0.0);
    float count = 1.0e-12;
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int nImChunk = get_global_size(0);
    const unsigned long int q = get_global_id(1);
    const unsigned long int rho = q/nTheta;
    const unsigned long int theta = q % nTheta;

    const unsigned long int indxStart = (theta + nTheta * rho)*indxstep;
    unsigned long int i, idx;

    for (i=0; i<indxstep; i++){
      idx = rdnIndx[indxStart+i];
      if (idx >= imstep) {
        break;
      } else {
        sum += images[idx*nImChunk + gid_im];
        count += 1.0;
      }
    }


    const unsigned long int rndIndx = (theta+thetaPad + (rho+rhoPad)*nThetaP)*nImChunk + gid_im;
    radon[rndIndx] = sum/count;
    //radon[rndIndx] = gid_im;
    }

  __kernel void radonFixArt(
      __global float16 *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int thetaPad)
  {
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int nImChunk = get_global_size(0);
    const unsigned long int rho = get_global_id(1);
    const unsigned long int rhoIndx = nTheta * rho;
    //rndIndx = nTheta * nRho * gid_im + (nTheta * rho);

    //radon[gid_rdn+thetaPad] = radon[gid_rdn+thetaPad+1];
    radon[(thetaPad + rhoIndx)*nImChunk + gid_im] = radon[(thetaPad + 1 + rhoIndx)*nImChunk + gid_im];
    //radon[gid_rdn+nTheta-1-thetaPad] = radon[gid_rdn+nTheta-2-thetaPad];
    radon[(nTheta-1-thetaPad + rhoIndx)*nImChunk + gid_im] = radon[(+nTheta-2-thetaPad + rhoIndx)*nImChunk + gid_im];
  }

// Padding of the radon Theta -- 0 and 180* are symmetric with a vertical flip.
__kernel void radonPadTheta(
      __global float *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int thetaPad)
  {
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int nImChunk = get_global_size(0);
    const unsigned long int gid_rho = get_global_id(1);
    unsigned long int gid_rdn1,gid_rdn2, i;
    gid_rdn1 =   (nTheta * gid_rho);
    gid_rdn2 =   (nTheta * (nRho -1 - gid_rho));

    for (i = 0; i <= thetaPad; ++i){
        radon[(i + gid_rdn1)*nImChunk + gid_im] = radon[ (gid_rdn2 + (nTheta-1 - (2 * thetaPad)) + i) * nImChunk + gid_im ];
        radon[ (gid_rdn2 + (nTheta-1 - ( thetaPad)) + i) * nImChunk + gid_im] = radon[ (gid_rdn1 + thetaPad + i)*nImChunk +gid_im];
    }

  }

// Padding of the radon Rho -- copy the defined line to all other rows ...
  __kernel void radonPadRho(
      __global float *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int rhoPad)
  {
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int nImChunk = get_global_size(0);
    const unsigned long int gid_theta = get_global_id(1);
    unsigned long int i, gid_rdn1, gid_rdn2;
    //indxim = nTheta * nRho * gid_im;
    //rd1p =  radon[indxim + (nTheta * rhoPad) + gid_theta] ;
    //rd2p =  radon[ indxim + (nTheta * (nRho -1 - rhoPad)) + gid_theta] ;
    const float rd1p =  radon[((nTheta * rhoPad) + gid_theta)*nImChunk + gid_im] ;
    const float rd2p =  radon[((nTheta * (nRho -1 - rhoPad)) + gid_theta)*nImChunk+gid_im];

    for (i = 0; i <= rhoPad; ++i){

        //gid_rdn1 = indxim + (nTheta*i) + gid_theta;
        //gid_rdn2 = indxim + (nTheta* (nRho-1-rhoPad+i)) + gid_theta;
        gid_rdn1 = ((nTheta*i) + gid_theta)*nImChunk + gid_im;
        gid_rdn2 = ((nTheta* (nRho-1-rhoPad+i)) + gid_theta)*nImChunk + gid_im;
        radon[gid_rdn1] = rd1p;
        radon[gid_rdn2] = rd2p;
    }

  }

// Convolution of a stack of images by a 2D kernel
// At somepoint we might want to consider the ability to chain together convolutions -- keeping the max at each pixel...
__kernel void convolution3d2d( __global const float16 *in, __constant float *kern, const int imszx, const int imszy, const int imszz, 
  const int kszx, const int kszy, const int padx, const int pady, __global float16 *out)
{
  // IDs of work-item represent x and y coordinates in image
  const long x = get_global_id(0) + padx;
  const long y = get_global_id(1) + pady;
  const unsigned long int z = get_global_id(2);
  const unsigned long int nImChunk = get_global_size(2);
  //long int indxIm; 
  //long int indxK;
  unsigned long yIndx; 
  unsigned long yKIndx; 
  float16 pxVal;
  float16 sum = (float16) (0.0f); // initialize convolution sum
  float kVal;
  const int kszx2 = kszx/2;
  const int kszy2 = kszy/2;

  const long int istart = ((x+kszx2) >= imszx) ? x+kszx2+1-imszx: 0;
  const long int iend = ((x-kszx2) < 0) ? x+kszx2+1: kszx;
  const long int startx = x+kszx2;
  const long int jstart = ((y+kszy2) >= imszy) ? y+kszy2+1-imszy: 0;
  const long int jend = ((y-kszy2) < 0) ? y+kszy2+1: kszy;
  const long int starty = y+kszy2;
  
  
  
  //const long int zIndx = imszx*imszy*z;
  //float current = out[(y*imszx + x) * nImChunk + z];

  for(int j=jstart; j<jend; ++j)
  {
      //yIndx = zIndx + (j)*imszx;
      //yKIndx = (j-y)*kszx;
      yKIndx = kszx*(j);
      yIndx = imszx*(starty - j);
      for(int i=istart; i<iend; ++i)
      {      
        pxVal = in[(yIndx+(startx - i)) * nImChunk + z];
        kVal = kern[yKIndx + i];
        sum += pxVal*kVal;
    }
  }
  // sum = current < sum ? sum : current;
  // write value of pixel to output image at location (x, y,z)
  out[(y*imszx + x) * nImChunk + z] = sum;
}

//Brute force gray-scale dilation (or local max).  This may be replaced with the van Herk/Gil-Werman algorithm
__kernel void morphDilateKernelBF( __global const float16 *in,  __global float16 *out,
                                   const long imszx, const long imszy, 
                                   const long padx, const long pady,
                                   const long kszx, const long kszy)
{
  //IDs of work-item represent x and y coordinates in image; z is the n-th image.
  //const long int x = get_global_id(0) + xpad;
  //const long int y = get_global_id(1) + ypad;
  const long x = get_global_id(0) + padx;
  const long y = get_global_id(1) + pady;
  const long int z = get_global_id(2);
  const long int nImChunk = get_global_size(2);
  long int yIndx;
  float16 pxVal;
  const  long kszx2 = kszx/2;
  const  long kszy2 = kszy/2;

  const long int istart = x-kszx2 >= 0 ? x-kszx2: 0;
  const long int iend = x+kszx2 < imszx ? x+kszx2: imszx-1;
  const long int jstart = y-kszy2 >= 0 ? y-kszy2: 0;
  const long int jend = y+kszy2 < imszy ? y+kszy2: imszy-1;

  // extreme pixel value (max/min for dilation/erosion) inside structuring element
  float16 extremePxVal = (float16) (-1.0e12f); // initialize extreme value with min when searching for max value


  for(int j=jstart; j<=jend; ++j)
  {
      yIndx = j*imszx;

      for(int i=istart; i<=iend; ++i)
      {

      pxVal = in[(yIndx+i)*nImChunk+z];
      //const float pxVal = read_imagef(in, sampler_dilate, (int3)(x + i, y + j, z)).s0;
      extremePxVal = max(extremePxVal, pxVal);
    }
  }

  // write value of pixel to output image at location (x, y,z)
  out[(y*imszx + x)*nImChunk+z] = extremePxVal;
}

//find the minimum value in each image.  This probably could be sped up by using a work group store.
__kernel void imageMin( __global const float16 *im1, __global float16 *imMin,
                        const unsigned int imszx, const unsigned int imszy, const unsigned int padx, const unsigned int pady)
  {
  const unsigned long int z = get_global_id(0);
  const unsigned long int nImChunk = get_global_size(0);
  long int indx,i, j;
  float16 cmin = (float16) (1.0e12);
  float16 imVal;

  //indxz = z*imszx*imszy;
  for(j = pady; j<= imszy - pady-1; ++j){
    indx = j*imszx;
    for(i = padx; i<= imszx - padx-1; ++i){
        imVal = im1[(indx+i)*nImChunk+z];
        cmin = select(cmin, imVal, (imVal < cmin));
    }   
  }
  imMin[z] = cmin;
}

// Subtract a value from an image stack, with clipping.
// The value to be subtracted are unique to each image, stored in an array in imMin
__kernel void imageSubMinWClip( __global float16 *im1, __global const float16 *imMin,
  const unsigned int imszx, const unsigned int imszy, 
  const unsigned int padx, const unsigned int pady)
  {
  // IDs of work-item represent x and y coordinates in image
  //const unsigned long int x = get_global_id(0) + padx;
  //const unsigned long int y = get_global_id(1) + pady;
  const long x = get_global_id(0) + padx;
  const long y = get_global_id(1) + pady;
  const unsigned long z = get_global_id(2);
  const unsigned long nImChunk = get_global_size(2);

  const long int indx = (x+y*imszx)*nImChunk + z;
  const float16 im1val = im1[indx];
  float16 value = im1val - imMin[z];
  
  //im1[indx] = (value < (float16) (0.0f)) ? (float16) (0.0f) : value;
  im1[indx] = select(value, (float16) (0.0f), (value < (float16) (0.0f)) );
}



// Is image1 (can be a stack of images) EQ to image2 (can be a stack) -- returns byte array.
// There is ability to include padding in x and y
__kernel void im1EQim2( __global const float *im1, __global const float *im2, __global uchar *out,
                        const unsigned long imszx, const unsigned long imszy, 
                        const unsigned long padx, const unsigned long pady)
{
  // IDs of work-item represent x and y coordinates in image
  //const unsigned long int x = get_global_id(0)+padx;
  //const unsigned long int y = get_global_id(1)+pady;
  const unsigned long x = get_global_id(0) + padx;
  const unsigned long y = get_global_id(1) + pady;
  const unsigned long int z = get_global_id(2);
  const unsigned long int nImChunk = get_global_size(2);
  //const int16 yes = (int16)(1);
  //const int16 no = (int16)(0);

  const unsigned long int indx = (x + imszx*y)*nImChunk + z;
  const float im1val = im1[indx];
  const float im2val = im2[indx];
  const float diff = fabs(im1val - im2val);
  //out[indx] = (diff < 1.0e-6f) ? yes: no;
  out[indx] = (diff < 1.0e-5f) ? 1: 0;
  
  
}

//this is a dirty little sort for getting the max locations ordered.  Will order accending.  
// Keeping only the top nmax points.  Need to have maxval1d primed so that maxval1d[nmax] is larger 
// than anything that will occur on its own.  
void dirtsort( __global float *maxval1d, __global long int *maxloc1d,
                long int newloc, float newmax, unsigned long nmax);

void dirtsort( __global float *maxval1d, __global long int *maxloc1d,
                long int newloc, float newmax, unsigned long nmax){

  unsigned long int i, istop;
  if (newmax < maxval1d[nmax-1]){
    return;
  }
  else{
    istop = 0; 
    for (i=1; i<nmax; ++i){
      if (newmax > maxval1d[nmax-1-i]){
        maxval1d[nmax-i] = maxval1d[nmax-1-i];
        maxloc1d[nmax-i] = maxloc1d[nmax-1-i];
        istop = i;
      }
      else{
        //maxval1d[i-1]  = newmax;
        //maxloc1d[i-1]  = newloc;
        break;
      }
    }
    maxval1d[nmax-1-istop] = newmax;
    maxloc1d[nmax-1-istop]  = newloc;
    return;
  }
}


// Is image1 (can be a stack of images)
__kernel void maxlabel( __global const uchar *maxlocin,__global const float *maxvalin,
                        __global long int *maxloc,__global float *maxval,
                        __global float2 *aveloc,__global float *aveval,
                        __global float *width,
                        const long imszx, const long imszy, 
                        const long padx, const long pady, const long nmax)
{
  // IDs of work-item represent x and y coordinates in image
  //const unsigned long x = get_global_id(0) + padx;
  //const unsigned long y = get_global_id(1) + pady;
  const unsigned long int z = get_global_id(0);
  const unsigned long int nImChunk = get_global_size(0);
  uchar imVal1;
  float imVal2, imValyp1, imValym1; 
  float dx, dy, dxx, dyy, dxy, det, w, ix, iy; 
  const long lnmax = (long) nmax; 
  long int indy, indxy,i, j; //,k;
  //__global long int maxloc1d[32];// = {0};
  //__global float maxval1d[32];// = {-1e12f};
  long int x,y;
  float  avetempweight;
  float2  avetempxy;



  // first find all the local max points identified and sort them
  for (i=0;i<lnmax;++i){
    maxloc[i+lnmax*z] = 0;
    maxval[i+lnmax*z] = -1.0e12f;
  }


  //maxval1d[lnmax] = 1.0e12f; // prime the pump for sorting 
  for(j = pady; j< imszy - pady; ++j){
    indy = j*imszx;
    for(i = padx; i< imszx - padx; ++i){
        indxy = indy+i;

        imVal1 = maxlocin[(indxy)*nImChunk+z];
        if (imVal1 == 0){
          continue;
        }
        else{
          imVal2 = maxvalin[(indxy)*nImChunk+z];
          
          dirtsort(&(maxval[z*lnmax]), &(maxloc[z*lnmax]), indxy, imVal2, lnmax);
        }
        
      
    }   
  }

  // now place them in the output arrays
  for (i=0; i< lnmax; ++i){
    if (maxval[z*lnmax+i] > -1.0e6){
      //maxval[z*lnmax + i] = maxval1d[lnmax-i-1];
      indxy = maxloc[z*lnmax+i];
      x =  ( indxy % imszx );
      y =  (indxy / imszx );
      //maxloc[z*lnmax + i] = (float2) (y,x);
      //maxloc[(z*lnmax + i)+1] = (float) y;
      avetempxy = (float2) (0.0);
      avetempweight = 1.0e-12;
      // // really basic 9-neighbor peak interpolation
      // for (k=-1; k<=1;++k){
      //   indy = (y+k)*imszx;
      //   for (j=-1; j<=1;++j){
      //     indxy = indy+(x+j);
      //     imVal2 = maxvalin[(indxy)*nImChunk+z];
      //     avetempxy += ((float2) (y+k, x+j))*imVal2;
      //     //avetempx += (x+j)*imVal2;
      //     avetempweight += imVal2;
      //   }
      // }
      // avetempxy /= avetempweight;
      // aveloc[z*lnmax + i] = avetempxy;
      // aveval[z*lnmax + i] = avetempweight/9.0;
      // // band width metric
      // imValyp1 = maxvalin[((y+1)*imszx+x)*nImChunk+z];
      // imValym1 = maxvalin[((y-1)*imszx+x)*nImChunk+z];
      // width[z*lnmax + i] = 1.0 / (maxval[z*lnmax+i] - 0.5 * (imValyp1 + imValym1) + 1e-12) ;

      // More advanced Taylor series approximation 
      dx = 0.0; dy = 0.0; dxx = 0.0; dyy = 0.0; dxy = 0.0; 
      indxy = ((y-1)*imszx + (x-1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxy += imVal2; 

      indxy = ((y-1)*imszx + (x))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dy  -= imVal2;
      dyy += imVal2;
      imValym1 = imVal2; 


      indxy = ((y-1)*imszx + (x+1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxy -= imVal2;

      indxy = ((y)*imszx + (x-1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dx -= imVal2; 
      dxx += imVal2; 

      indxy = ((y)*imszx + (x))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxx -= 2.0 * imVal2;
      dyy -= 2.0 * imVal2;
      w = imVal2;

      indxy = ((y)*imszx + (x+1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dx  +=  imVal2;
      dxx +=  imVal2;
      
      indxy = ((y+1)*imszx + (x-1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxy -=  imVal2;
      
      indxy = ((y+1)*imszx + (x))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dy  +=  imVal2;
      dyy += imVal2; 
      imValyp1 = imVal2;

      indxy = ((y+1)*imszx + (x+1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxy  +=  imVal2;

      dxy *= 0.25;
      dx  *= 0.5;
      dy  *= 0.5;

      det = (dxx*dyy - dxy*dxy);
      det = (fabs(det) > 1.0e-12) ? det : 1.0e-12; 
      det = 1.0 / det; 

      ix = (float) x - (dyy * dx - dxy * dy) * det; 
      iy = (float) y - (dxx * dy - dxy * dx) * det; 

      aveloc[z*lnmax + i] = (float2) (iy, ix); 
      aveval[z*lnmax + i] = avetempweight/9.0;
      // band width metric
      width[z*lnmax + i] = 1.0 / (w - 0.5 * (imValyp1 + imValym1) + 1e-12) ;

    }
    else{
      break; // no more detected peaks
    }

  }
}














// These never quite worked.  Not the biggest time penalty anyway. 

void morphDilateKernel(const unsigned long winsz, const unsigned long winsz2,
                      __local float16 *s,  __local float16 *r,  __local float16 *w);

void morphDilateKernel(const unsigned long winsz, const unsigned long winsz2,
                      __local float16 *s,  __local float16 *r,  __local float16 *w){
  unsigned long int i;
  const unsigned long  winsz1 = winsz-1;

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

__kernel void morphDilateKernelX( __global const float16 *im1, __global float16 *out,
                        const unsigned long winszX, const unsigned long winszX2,
                        const unsigned long imszx, const unsigned long imszy,
                        const unsigned long padx, const unsigned long pady,
                        __local float16 *s, __local float16 *r, __local float16 *w)
{
  // IDs of work-item represent x and y coordinates in image
  const unsigned long int chunknum = get_global_id(0);
  const unsigned long int y = get_global_id(1)+pady;
  const unsigned long int z = get_global_id(2);
  const unsigned long int nImChunk = get_global_size(2);
  
  unsigned long int i;
  unsigned long int indx;


  for (i=0; i<winszX; i++){
    s[i] = (float16) (-1.0e12f);
    r[i] = (float16) (-1.0e12f);
  }

  for (i =0; i<(winszX*2-1); ++i){
    indx = ((padx + chunknum*winszX - winszX2 +i)  + imszx*y)*nImChunk + z;
    w[i] = im1[indx];
  }

  morphDilateKernel(winszX, winszX2,s, r,w);
  for (i =winszX2; i<(winszX2+winszX); ++i){
    indx = ((padx + chunknum*winszX - winszX2 +i)  + imszx*y)*nImChunk + z;
    out[indx] = w[i];
  }

}


__kernel void morphDilateKernelY( __global const float16 *im1, __global float16 *out,
                        const unsigned long winszY, const unsigned long winszY2,
                        const unsigned long imszx, const unsigned long imszy,
                        const unsigned long padx, const unsigned long pady,
                        __local float16 *s, __local float16 *r, __local float16 *w)
{
  // IDs of work-item represent x and y coordinates in image
  const unsigned long int x = get_global_id(0) + padx;
  const unsigned long int chunknum = get_global_id(1);
  const unsigned long int z = get_global_id(2);
  const unsigned long int nImChunk = get_global_size(2);

  //float s[128];
  //float r[128];
  //float w[256];


  unsigned long int i;
  unsigned long int indx; ;

  for (i=0; i<winszY; i++){
    s[i] = -1.0e12;
    r[i] = -1.0e12;
  }

  for (i =0; i<(winszY*2-1); ++i){
    indx = (x + (pady + chunknum*winszY - winszY2 +i) * imszx)*nImChunk + z;
    w[i] = im1[indx];
  }

  morphDilateKernel(winszY, winszY2,s, r,w);
  for (i =winszY2; i<(winszY2+winszY); ++i){
    indx = (x + (pady + chunknum*winszY - winszY2 +i) * imszx)*nImChunk + z;
    out[indx] = w[i];
  }

}



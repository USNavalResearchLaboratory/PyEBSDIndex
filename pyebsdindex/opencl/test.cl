

float sum16(const float16 v1);

// float sum16(const float16 *v1){
//  float sum = 0.0;
//  sum += v1[0].s0;
//         sum += v1[0].s1;
//         sum += v1[0].s2;
//         sum += v1[0].s3;
//         sum += v1[0].s4;
//         sum += v1[0].s5;
//         sum += v1[0].s6;
//         sum += v1[0].s7;
//         sum += v1[0].s8;
//         sum += v1[0].s9;
//         sum += v1[0].sa;
//         sum += v1[0].sb;
//         sum += v1[0].sc;
//         sum += v1[0].sd;
//         sum += v1[0].se;
//         sum += v1[0].sf;
//      return sum;

// }


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



__kernel void testdot(float t){
printf("%f\n",t ); 
float16 d1 = (float16)(1.0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
//float16 d2 = (float16)(17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32);
//float8 d11 = d1.lo;
//float8 d22 = d2.lo;
//float t;


//float8 d1 = (float8) (1,2,3,4,4,3,2,1);
//float8 d2 = (float8) (5,6,7,8,5,6,7,8);
//t = d11.s0;
t = sum16(d1);
printf("%f",t ); 

}

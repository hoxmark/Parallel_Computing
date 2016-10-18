#include <complex.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>

/*
mmintrin.h - MMX
xmmintrin.h - SSE
emmintrin.h - SSE2
pmmintrin.h - SSE3
smmintrin.h - SSE4.1
nmmintrin.h - SSE4.2
immintrin.h - AVX
*/

//Currently just a copy of naive... + A*alpha outisde
void chemm(complex float* A,
        complex float* B,
        complex float* C,
        int m,
        int n,
        complex float alpha,
        complex float beta){

    complex* matrix = malloc(m*m*sizeof ( complex));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            matrix[j*m+i] = alpha*A[j*m+i];
        }
    }

    for(int x = 0; x < n; x++){
        for(int y = 0; y < m; y++){
            C[y*n + x] *= beta;
            for(int z = 0; z < m; z+=4){
                //OLD:
                // C[y*n + x] += matrix[y*m+z]*B[z*n + x];
                
                //making sets of 4, so we can do one instruction on sets of 4 data at a time.         
                __m128 complex_real = _mm_set_ps(crealf(matrix[y*m+z+3]), crealf(matrix[y*m+z+2]), crealf(matrix[y*m+z+1]), crealf(matrix[y*m+z])); // x
                __m128 complex_imag = _mm_set_ps(cimagf(matrix[y*m+z+3]), cimagf(matrix[y*m+z+2]), cimagf(matrix[y*m+z+1]), cimagf(matrix[y*m+z])); // y
                __m128 real_b = _mm_set_ps(crealf(B[(z+3)*n + x]), crealf(B[(z+2)*n + x]), crealf(B[(z+1)*n + x]), crealf(B[z*n + x])); // u
                __m128 imag_b = _mm_set_ps(cimagf(B[(z+3)*n + x]), cimagf(B[(z+2)*n + x]), cimagf(B[(z+1)*n + x]), cimagf(B[z*n + x])); // v

                //calculate ac
                __m128 ansAC = _mm_mul_ps(complex_real, real_b);
                //calculate yv 
                __m128 ansYV = _mm_mul_ps(complex_imag, imag_b);
                //calcualte ac - yv = real
                float newReal[4]; 
                __m128 ansReal = _mm_sub_ps(ansAC, ansYV);
                _mm_store_ps(newReal, ansReal); 

                //calculate xv
                __m128 ansxv = _mm_mul_ps(complex_real, imag_b);
                //calculate yu 
                __m128 ansyu = _mm_mul_ps(complex_imag, real_b);
                //calcualte xv + yu = imag
                float newImag[4]; 
                __m128 ansimag = _mm_add_ps(ansxv, ansyu);
                _mm_store_ps(newImag, ansimag); 

                //back to complex: 
                complex c1 = newReal[0] + newImag[0] * I;
                complex c2 = newReal[1] + newImag[1] * I;
                complex c3 = newReal[2] + newImag[2] * I;
                complex c4 = newReal[3] + newImag[3] * I;

                C[y*n + x] += c1;
                C[y*n + x] += c2;
                C[y*n + x] += c3;
                C[y*n + x] += c4;

            }
        }
    }

    free(matrix);
}

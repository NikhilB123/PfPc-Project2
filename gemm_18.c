#include "driver.h"
#include<immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#define alpha( i,j ) A[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A
#define beta( i,j )  B[ (j)*ldB + (i) ]   // map beta( i,j ) to array B
#define gamma( i,j ) C[ (j)* ldC + (i) ]   // map gamma( i,j ) to array C

void LoopFive( int, int, int, double *, int, double *, int, double *, int );
void LoopFour( int, int, int, double *, int, double *, int,  double *, int );
void LoopThree( int, int, int, double *, int, double *, double *, int );
void LoopTwo( int, int, int, double *, double *,  double *, int );
void LoopOne( int, int, int, double *, double *, double *, int );
void Gemm_MRxNRKernel_Packed( int, double *, double *, double *, int );
void PackBlockA( int, int, double *, int, double * );
void PackMicroPanelA_MRxKC( int, int, double *, int, double *);
void PackMicroPanelB_KCxNR( int, int, double *, int, double *);

void PackPanelB( int, int, double *, int, double * );

/* Blocking parameters */
#define MC 12
#define MR 1
#define NC 4
#define NR 1
#define KC 4
  
/* Wrapper for GEMM function */
void MyGemm( int m, int n, int k, double *A, int ldA,
	     double *B, int ldB, double *C, int ldC )
{


  if (MC == -1 || MR == -1 || NC == -1 || NR == -1 || KC == -1 )
  {
    printf("Some of the blocking parameters are not set\n");
    exit(0);
  }
  if ( m % MR != 0 || MC % MR != 0 ){
    printf( "m and MC must be multiples of MR\n" );
    exit( 0 );
  }
  if ( n % NR != 0 || NC % NR != 0 ){
    printf( "n and NC must be multiples of NR\n" );
    exit( 0 );
  }

  LoopFive( m, n, k, A, ldA, B, ldB, C, ldC );
}

void LoopFive(  int m, int n, int k, 
                double *A, int ldA,
		            double *B, int ldB, 
                double *C, int ldC )
{
  for ( int j=0; j<n; j+=NC ) 
  {
    int jb = dmin( NC, n-j );    /* Last loop may not involve a full block */
    LoopFour( m, jb, k, A, ldA, &beta( 0,j ), ldB, &gamma( 0,j ), ldC );
  } 
}

void LoopFour(  int m, int n, int k, 
                double *A, int ldA, 
                double *B, int ldB,
	              double *C, int ldC )
{
  double *Btilde = ( double * ) _mm_malloc( KC * NC * sizeof( double ), 64 );
  
  for ( int p=0; p<k; p+=KC ) 
  {
    int pb = dmin( KC, k-p );    /* Last loop may not involve a full block */
    PackPanelB( pb, n, &beta( p, 0 ), ldB, Btilde );
    LoopThree( m, n, pb, &alpha( 0, p ), ldA, Btilde, C, ldC );
  }

  _mm_free( Btilde); 
}

void LoopThree( int m, int n, int k, 
                double *A, int ldA, 
                double *Btilde, 
                double *C, int ldC )
{
  double *Atilde = ( double * ) _mm_malloc( MC * KC * sizeof( double ), 64 );
       
  for ( int i=0; i<m; i+=MC ) {
    int ib = dmin( MC, m-i );    /* Last loop may not involve a full block */
    PackBlockA( ib, k, &alpha( i, 0 ), ldA, Atilde );
    LoopTwo( ib, n, k, Atilde, Btilde, &gamma( i,0 ), ldC );
  }

  _mm_free( Atilde);
}

void LoopTwo( int m, int n, int k, 
              double *Atilde, 
              double *Btilde, 
              double *C, int ldC )
{
  for ( int j=0; j<n; j+=NR ) {
    int jb = dmin( NR, n-j );
    LoopOne( m, jb, k, Atilde, &Btilde[ j*k ], &gamma( 0,j ), ldC );
  }
}

void LoopOne( int m, int n, int k, 
              double *Atilde, 
              double *MicroPanelB, 
              double *C, int ldC )
{
  for ( int i=0; i<m; i+=MR ) {
    int ib = dmin( MR, m-i );
    Gemm_MRxNRKernel_Packed( k, &Atilde[ i*k ], MicroPanelB, &gamma( i,0 ), ldC );
  }
}


/* DGEMM mircokernel 
  Computes C += AB where C is MR x NR, A is MR x KC, and B is KC x NR */
void Gemm_MRxNRKernel_Packed( int k, double *A,double *B,
		double *C, int ldC )
{
  __m256d gamma_0123_0 = _mm256_loadu_pd( &gamma( 0,0 ) );
  __m256d gamma_0123_1 = _mm256_loadu_pd( &gamma( 0,1 ) );
  __m256d gamma_0123_2 = _mm256_loadu_pd( &gamma( 0,2 ) );
  __m256d gamma_0123_3 = _mm256_loadu_pd( &gamma( 0,3 ) );

  __m256d beta_p_j;
   	
  for ( int p=0; p<k; p++ ){
    /* load alpha( 0:3, p ) */
    __m256d alpha_0123_p = _mm256_loadu_pd( A );

    /* load beta( p, 0 ); update gamma( 0:3, 0 ) */
    beta_p_j = _mm256_broadcast_sd( B );
    gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );

    /* load beta( p, 1 ); update gamma( 0:3, 1 ) */
    beta_p_j = _mm256_broadcast_sd( B+1 );
    gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_1 );

    /* load beta( p, 2 ); update gamma( 0:3, 2 ) */
    beta_p_j = _mm256_broadcast_sd( B+2 );
    gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_2 );

    /* load beta( p, 3 ); update gamma( 0:3, 3 ) */
    beta_p_j = _mm256_broadcast_sd( B+3 );
    gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_3 );

    A += MR;
    B += NR;
  }

  /* Store the updated results.  This should be done more carefully since
     there may be an incomplete micro-tile. */
  _mm256_storeu_pd( &gamma(0,0), gamma_0123_0 );
  _mm256_storeu_pd( &gamma(0,1), gamma_0123_1 );
  _mm256_storeu_pd( &gamma(0,2), gamma_0123_2 );
  _mm256_storeu_pd( &gamma(0,3), gamma_0123_3 );
}

/* Pack a MC x KC block of A into Atilde */
void PackBlockA( int m, int k, double *A, int ldA, double *Atilde )
   //This is an unoptimized implementation for general MR and KC. */
{
  /* March through A in column-major order, packing into Atilde as we go. */

  if ( m == MR ) {
    /* Full row size micro-panel.*/
    for ( int p=0; p<k; p++ ) 
      for ( int i=0; i<MR; i++ ) 
        *Atilde++ = alpha( i, p );
  }
  else {
    /* Not a full row size micro-panel.  We pad with zeroes.  To be  added */
  }
}

void PackMicroPanelA_MRxKC( int m, int k, double *A, int ldA, double *Atilde ) 
/* Pack a micro-panel of A into buffer pointed to by Atilde. 
   This is an unoptimized implementation for general MR and KC. */
{
  /* March through A in column-major order, packing into Atilde as we go. */

  if ( m == MR ) {
    /* Full row size micro-panel.*/
    for ( int p=0; p<k; p++ ) 
      for ( int i=0; i<MR; i++ ) 
        *Atilde++ = alpha( i, p );
  }
  else {
    /* Not a full row size micro-panel.  We pad with zeroes.  To be  added */
  }
}

/* Pack a KC x NC block of B into Btilde */
void PackPanelB( int k, int n, double *B, int ldB, double *Btilde )
/* Pack a k x n panel of B in to a KC x NC buffer.
.  
   The block is copied into Btilde a micro-panel at a time. */
{
  for ( int j=0; j<n; j+= NR ){
    int jb = dmin( NR, n-j );
    
    PackMicroPanelB_KCxNR( k, jb, &beta( 0, j ), ldB, Btilde );
    Btilde += k * jb;
  }
}

void PackMicroPanelB_KCxNR( int k, int n, double *B, int ldB,
	    double *Btilde )
/* Pack a micro-panel of B into buffer pointed to by Btilde.
   This is an unoptimized implementation for general KC and NR.
   k is assumed to be less then or equal to KC.
   n is assumed to be less then or equal to NR.  */
{
  /* March through B in row-major order, packing into Btilde. */
  if ( n == NR ) {
    /* Full column width micro-panel.*/
    for ( int p=0; p<k; p++ )
      for ( int j=0; j<NR; j++ )
        *Btilde++ = beta( p, j );
  }
  else {
    /* Not a full row size micro-panel. We pad with zeroes.
     To be added */
  }
}
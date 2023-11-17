#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include<math.h>

void conjugategradient_p(int *A, int *b, double *x, int n);
void conjugategradient_s(int *A, int *b, double *x, int n);

int main(int argc, char *argv[])
{
    long n;
    //input matrix dimensions
    printf("Enter matrix dimension for a square matrix: ");
    n = 3;

    //defining requirements
    int *matrix = (int *)malloc(sizeof(int) * n * n);
    int *b = (int *)malloc(sizeof(int) * n);
    double *x = (double *)malloc(sizeof(double) * n);
    matrix[0] = 10;
    matrix[1] = -1;
    matrix[2] = 3;
    matrix[3] = 1;
    matrix[4] = 11;
    matrix[5] = -5;
    matrix[6] = 2;
    matrix[7] = -1;
    matrix[8] = 13;
    b[0] = 7;
    b[1] = 5;
    b[2] = 8;

    // matrices
    int *matrix3 = (int *)malloc(sizeof(int) * n * n);
    int *b3 = (int *)malloc(sizeof(int) * n);
    double *x3 = (double *)malloc(sizeof(double) * n);
    int *matrix4 = (int *)malloc(sizeof(int) * n * n);
    int *b4 = (int *)malloc(sizeof(int) * n);
    double *x4 = (double *)malloc(sizeof(double) * n);

    for(int i =0; i<n*n; i++)
    {
        matrix3[i] = matrix[i];
        matrix4[i] = matrix[i];
    }

    for(int i = 0; i<n ; i++)
    {
        b3[i] = b[i];
        b4[i] = b[i];
    }


    printf("\n\nA * x = b  \n");
    printf("To find : x \n\n");

    //A Matrix
    printf("Matrix A: \n");
    for(int i = 0; i<n*n; i++)
    {
        printf("%d   ",matrix[i]);
        if((i+1)%n == 0)
        {
            printf("\n");
        }
    }

    //B Matrix
    printf("\n\nMatrix b: \n");
    for(int i = 0; i<n; i++)
    {
        printf("%d\n", b[i]);
    }

    printf("\n\nFINDING SOLUTIONS: \n");

    //Conjugate Gradient
    printf("\n 2. Conjugate Gradient: \n");
    printf("\t A. Serially: \n");
    double tb = omp_get_wtime();
    conjugategradient_s(matrix3, b3, x3, n);
    tb = omp_get_wtime() - tb;
    printf("\t B. Parallely: \n");
    double tb1 = omp_get_wtime();
    conjugategradient_p(matrix4,b4,x4,n);
    tb1 = omp_get_wtime() - tb1;

    printf("\n\t Time for serial execution: %0.30f\n\n", tb);
    printf("\n\t Time for parallel execution: %0.30f\n\n", tb1);

    free(matrix);
    free(b);
    free(x);
    free(matrix3);
    free(b3);
    free(x3);
    free(matrix4);
    free(b4);
    free(x4);
    return 0;
}

void conjugategradient_p(int *A, int *b, double *x,  int n)
{
    int t;
    int max_iterations;
    printf("\nEnter number of iterations: ");
    scanf("%d", &max_iterations);

    printf("\nEnter number of threads: ");
    scanf("%d", &t);

    double r[n];
    double p[n];
    double px[n];

    #pragma omp parallel for num_threads(t)
    for( int i = 0 ; i<n ; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }


    int q = max_iterations;

    double alpha = 0;

    while(q--)
    {
        double sum = 0;
        #pragma omp parallel  for num_threads(t)  reduction(+ : sum)
        for(int i = 0 ; i < n ; i++)
        {
            sum = r[i]*r[i] + sum;
        }

        double temp[n];
        #pragma omp parallel for num_threads(t)
        for( int i = 0; i<n ; i++ )
        {
            temp[i] = 0;
        }

        double num = 0;
        #pragma omp parallel for num_threads(t)
        for(int i = 0 ; i < n ; i++)
        {
            #pragma omp parallel for reduction(+ : temp[i])
            for(int j = 0 ; j < n ; j++ )
            {
                temp[i] = A[i*n+j]*p[j] + temp[i];
            }
        }
        #pragma omp parallel for num_threads(t) reduction(+ : num)
        for(int j = 0 ; j < n ; j++)
        {
            num = num + temp[j]*p[j];
        }

        alpha = sum / num;

        #pragma omp parallel for num_threads(t)
        for(int i = 0; i < n ; i++ )
        {
            px[i] = x[i];
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*temp[i];
        }

        double beta = 0;
        #pragma omp parallel for num_threads(t) reduction(+ : beta)
        for(int i = 0 ; i < n ; i++)
        {
            beta = beta + r[i]*r[i];
        }

        beta = beta / sum;

        #pragma omp parallel for num_threads(t)
        for (int i = 0 ; i < n ; i++ )
        {
            p[i] = r[i] + beta*p[i];
        }

        int c=0;
        for(int i = 0 ; i<n ; i++ )
        {
            if(r[i]<0.000001)
                c++;
        }

        if(c==n)
            break;

        }

    for( int i = 0 ; i<n ; i++ )
        printf("\t\t%f\n", x[i]);

    return;
}


void conjugategradient_s(int *A, int *b, double *x,  int n)
{
    int max_iterations;
    printf("\nEnter number of iterations: ");
    scanf("%d", &max_iterations);
    double r[n];
    double p[n];
    double px[n];
    for( int i = 0 ; i<n ; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }


    double alpha = 0;
    while(max_iterations--)
    {

        double sum = 0;
        for(int i = 0 ; i < n ; i++)
        {
            sum = r[i]*r[i] + sum;
        }

        double temp[n];
        for( int i = 0; i<n ; i++ )
        {
            temp[i] = 0;
        }

        double num = 0;
        for(int i = 0 ; i < n ; i++)
        {
            for(int j = 0 ; j < n ; j++ )
            {
                temp[i] = A[i*n+j]*p[j] + temp[i];
            }
        }
        for(int j = 0 ; j < n ; j++)
        {
            num = num + temp[j]*p[j];
        }

        alpha = sum / num;
        for(int i = 0; i < n ; i++ )
        {
            px[i] = x[i];
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*temp[i];
        }
        double beta = 0;
        for(int i = 0 ; i < n ; i++)
        {
            beta = beta + r[i]*r[i];
        }
        beta = beta / sum;
        for (int i = 0 ; i < n ; i++ )
        {
            p[i] = r[i] + beta*p[i];
        }
        int c=0;
        for(int i = 0 ; i<n ; i++ )
        {
            if(r[i]<0.000001 )
                c++;
        }
        if(c==n)
            break;
    }
    for( int i = 0 ; i<n ; i++ )
        printf("\t\t%f\n", x[i]);

    return;
}

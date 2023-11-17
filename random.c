#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include<math.h>

//A: Matrix of the system of linear equations.
//b: Right-hand side vector.
//x: Solution vector.
//r: Residual vector.
//p: Search direction vector.
//px: Temporary array to store the product of A and p.
//alpha, beta: Scalars used in the CG method.

int conjugategradient_p(double *A, double *b, double *x, int n);
int conjugategradient_s(double *A, double *b, double *x, int n);

int main(int argc, char *argv[])
{
    long n;
    //input matrix dimensions
    printf("Enter matrix dimension for a square matrix: ");
    scanf("%ld", &n);

    //defining requirements
    double *matrix = (double *)malloc(sizeof(double)*n*n);
    double *b = (double *)malloc(sizeof(double)*n);
    double *x = (double *)malloc(sizeof(double)*n);

    for(int i = 0; i< (n*n) ; i++)
    {
        double x = (rand()/(double)RAND_MAX);
        matrix[i] = x;
    }

    for(int i = 0; i<n; i++)
    {
        double x = (rand()/(double)RAND_MAX);
        b[i]= x;
    }

    // matrices
    double *matrix3 = (double *)malloc(sizeof(double)*n*n);
    double *b3 = (double *)malloc(sizeof(double)*n);
    double *x3 = (double *)malloc(sizeof(double)*n);

    double *matrix4 = (double *)malloc(sizeof(double)*n*n);
    double *b4 = (double *)malloc(sizeof(double)*n);
    double *x4 = (double *)malloc(sizeof(double)*n);


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
        printf("%f ",matrix[i]);
        if((i+1)%n == 0)
        {
            printf("\n");
        }
    }

    //B Matrix
    printf("\n\nMatrix b: \n");
    for(int i = 0; i<n; i++)
    {
        printf("%f\n", b[i]);
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

    printf("\n\t Time for serial execution: %0.2f\n\n", tb);
    printf("\n\t Time for parallel execution: %0.2f\n\n", tb1);

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

int conjugategradient_p(double *A, double *b, double *x,  int n)
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
        //Calculating dot product of resudual vector r --> r.r
        //Time Complexity : O((logn)/t)
        double sum = 0;
        #pragma omp parallel  for num_threads(t) reduction(+ : sum)
        for(int i = 0 ; i < n ; i++)
        {
            sum = r[i]*r[i] + sum;
        }
        
        
        //Assigning 0 to temp
        //temp will store the product of A and p
        //Time Complexity: O(n/t)
        double temp[n];
        #pragma omp parallel for num_threads(t)
        for( int i = 0; i<n ; i++ )
        {
            temp[i] = 0;
        }

        //Calculating pAp
        //Time Complexity: O(n/t)
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

        //Time Complexity: O(n/t)
        #pragma omp parallel for num_threads(t) reduction(+ : num)
        
        for(int j = 0 ; j < n ; j++)
        {
            num = num + temp[j]*p[j];
        }

        //Calculating Alpha
        alpha = sum / num;

        //Updating solution vector, residual vector and product vector px
        #pragma omp parallel for num_threads(t)
        //Time Complexity: O(n/t)
        for(int i = 0; i < n ; i++ )
        {
            px[i] = x[i];
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*temp[i];
        }

        
        //Calculating beta value
        double beta = 0;
        //Time Complexity: O(n/t)
        #pragma omp parallel for num_threads(t) reduction(+ : beta)
        for(int i = 0 ; i < n ; i++)
        {
            beta = beta + r[i]*r[i];
        }

        beta = beta / sum;

        //Updating search direction
        //Time Complexity: O(n/t) 
        #pragma omp parallel for num_threads(t)
        for (int i = 0 ; i < n ; i++ )
        {
            p[i] = r[i] + beta*p[i];
        }


        //Check for convergence by counting the number of elements in r close to zero.
        int c=0;

        for(int i = 0 ; i<n ; i++ )
        {
            if(r[i]<0.000001)
                c++;
        }

        //If all elements in r are close to zero, break out of the iteration loop.
        if(c==n)
            break;
    }

    for( int i = 0 ; i<n ; i++ )
        printf("\t\t%f\n", x[i]);

    return 0;
}


int conjugategradient_s(double *A, double *b, double *x,  int n)
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
        //for calculating dot product of resudual vector
        double sum = 0;

        for(int i = 0 ; i < n ; i++)
        {
            sum = r[i]*r[i] + sum;
        }
        
        //Assigning 0 to temp
        //temp will store the product of A and p
        double temp[n];

        for( int i = 0; i<n ; i++ )
        {
            temp[i] = 0;
        }

        //Calculating pAp
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

        //Calculating Alpha
        alpha = sum / num;

        //Updating solution vector, residual vector and product vector px
        for(int i = 0; i < n ; i++ )
        {
            px[i] = x[i];
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*temp[i];
        }

        //Calculating beta value
        double beta = 0;

        for(int i = 0 ; i < n ; i++)
        {
            beta = beta + r[i]*r[i];
        }
        beta = beta / sum;
        
        //Updating search direction
        for (int i = 0 ; i < n ; i++ )
        {
            p[i] = r[i] + beta*p[i];
        }

        //Check for convergence by counting the number of elements in r close to zero.
        
        int c=0;
        
        for(int i = 0 ; i<n ; i++ )
        {
            if(r[i]<0.000001 )
                c++;
        }
        
        //If all elements in r are close to zero, break out of the iteration loop.
        if(c==n)
            break;
    }
    for( int i = 0 ; i<n ; i++ )
        printf("\t\t%f\n", x[i]);

    return 0;
}



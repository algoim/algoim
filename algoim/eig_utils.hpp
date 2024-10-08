#ifndef ALGOIM_EIG_UTILS_HPP
#define ALGOIM_EIG_UTILS_HPP

// algoim::eig implements eigenvalue/vector related calculations as the SVD or the calculation 
// of Bernstein polynomials roots using a generalized eigenvalue problem.

#include "real.hpp"
#include "sparkstack.hpp"
#include "xarray.hpp"
#include "binomial.hpp"

#include <iostream>
#include <cassert>

#if WITH_LAPACK

// Some methods may rely on a LAPACK implementation to solve
// generalised eigenvalue problems and SVD factorisation
#if __has_include(<lapacke.h>)
#include <lapacke.h>
#elif __has_include(<mkl_lapacke.h>)
#include <mkl_lapacke.h>
#else
#error "WITH_LAPACK directive is active, then Algoim requires a LAPACKE implementation to compute eigenvalues and SVD factorisations, but a suitable lapacke.h include file was not found; did you forget to specify its include path?"
#endif

#endif // WITH_LAPACK

namespace algoim::bernstein
{

namespace detail
{

    /// Returns the value of a with the sign of b.
    static real sign(real a, real b)
    {
        return b < 0 ? -a : a;
    }


    /// Calculates sqrt( a^2 + b^2 ) with decent precision.
    /// Copied from https://gist.github.com/sasekazu/32f966816ad6d9244259
    static real pythag(const real a, const real b)
    {
        const real absa = std::abs(a);
        const real absb = std::abs(b);
        
        if (absa > absb)
            return (absa * std::sqrt(1.0 + std::pow(absb / absa, 2)));
        else
            return (absb == 0.0 ? 0.0 : absb * std::sqrt(1.0 + std::pow(absa / absb, 2)));
    }

    /**
     * @brief An implementation of SVD from Numerical Recipes in C and Mike Erhdmann's lectures
     * 
     * Function adapted from https://gist.github.com/sasekazu/32f966816ad6d9244259
     * (based on http://palantir.cs.colby.edu/maxwell/classes/e27/F03/labs/lab04/svdcmp.c)
     * 
     * Given a matrix a[nRows][nCols], svdcmp() computes its singular value
     * decomposition, A = U * W * Vt.  A is replaced by U when svdcmp
     * returns.  The diagonal matrix W is output as a vector w[nCols].
     * Vt is output as the matrix Vt[nCols][nCols].
     */
    static bool svdcmp(real **a, int nRows, int nCols, real *w, real **vt)
    {
        assert(0 < nCols && nCols <= nRows);

        /// Maximum number of iterations.
        static const int max_its = 30;

        int flag, i, its, j, jj, k, l, nm;
        real anorm, c, f, g, h, s, scale, x, y, z;

        std::vector<real> rv1;
        try
        {
            rv1.resize(nCols);
        }
        catch (const std::bad_alloc &ba) 
        {
            std::cerr << "algoim::svd::SVD::svdcmp() ERROR: " << ba.what() << std::endl;
            return false;
        }

        g = scale = anorm = 0.0;
        for (i = 0; i < nCols; i++)
        {
            l = i + 1;
            rv1[i] = scale * g;
            g = s = scale = 0.0;
            if (i < nRows)
            {
                for (k = i; k < nRows; k++)
                    scale += std::abs(a[k][i]);
                if (scale)
                {
                    for (k = i; k < nRows; k++)
                    {
                        a[k][i] /= scale;
                        s += a[k][i] * a[k][i];
                    }
                    f = a[i][i];
                    g = -detail::sign(std::sqrt(s),f);
                    h = f * g - s;
                    a[i][i] = f - g;
                    for (j = l; j < nCols; j++) 
                    {
                        for (s = 0.0, k = i; k < nRows; k++)
                            s += a[k][i] * a[k][j];
                        f = s / h;
                        for (k = i; k < nRows; k++)
                            a[k][j] += f * a[k][i];
                    }
                    for (k = i; k < nRows; k++)
                        a[k][i] *= scale;
                }
            }
            w[i] = scale * g;
            g = s = scale = 0.0;
            if (i < nRows && i != nCols - 1)
            {
                for (k = l; k < nCols; k++)
                    scale += std::abs(a[i][k]);
                if (scale) {
                    for (k = l; k < nCols; k++)
                    {
                        a[i][k] /= scale;
                        s += a[i][k] * a[i][k];
                    }
                    f = a[i][l];
                    g = -detail::sign(std::sqrt(s),f);
                    h = f * g - s;
                    a[i][l] = f - g;
                    for (k = l; k < nCols; k++)
                        rv1[k] = a[i][k] / h;
                    for (j = l; j < nRows; j++)
                    {
                        for (s = 0.0, k = l; k < nCols; k++)
                            s += a[j][k] * a[i][k];
                        for (k = l; k < nCols; k++)
                            a[j][k] += s * rv1[k];
                    }
                    for (k = l; k < nCols; k++)
                        a[i][k] *= scale;
                }
            }
            anorm = std::max(anorm, (std::abs(w[i]) + std::abs(rv1[i])));
        }

        for (i = nCols - 1; i >= 0; i--)
        {
            if (i < nCols - 1)
            {
                if (g)
                {
                    for (j = l; j < nCols; j++)
                        vt[i][j] = (a[i][j] / a[i][l]) / g;
                    for (j = l; j < nCols; j++)
                    {
                        for (s = 0.0, k = l; k < nCols; k++)
                            s += a[i][k] * vt[j][k];
                        for (k = l; k < nCols; k++)
                            vt[j][k] += s * vt[i][k];
                    }
                }
                for (j = l; j < nCols; j++)
                    vt[j][i] = vt[i][j] = 0.0;
            }
            vt[i][i] = 1.0;
            g = rv1[i];
            l = i;
        }

        for (i = std::min(nRows,nCols) - 1; i >= 0; i--)
        {
            l = i + 1;
            g = w[i];
            for (j = l; j < nCols; j++)
                a[i][j] = 0.0;
            if (g) {
                g = 1.0 / g;
                for (j = l; j < nCols; j++)
                {
                    for (s = 0.0, k = l; k < nRows; k++)
                        s += a[k][i] * a[k][j];
                    f = (s / a[i][i]) * g;
                    for (k = i; k < nRows; k++)
                        a[k][j] += f * a[k][i];
                }
                for (j = i; j < nRows; j++)
                    a[j][i] *= g;
            }
            else
            {
                for (j = i; j < nRows; j++)
                    a[j][i] = 0.0;
            }
            ++a[i][i];
        }

        for (k = nCols - 1; k >= 0; k--)
        {
            for (its = 0; its < max_its; its++)
            {
                flag = 1;
                for (l = k; l >= 0; l--)
                {
                    nm = l - 1;
                    if ((std::abs(rv1[l]) + anorm) == anorm) 
                    {
                        flag = 0;
                        break;
                    }
                    if ((std::abs(w[nm]) + anorm) == anorm)
                        break;
                }
                if (flag)
                {
                    c = 0.0;
                    s = 1.0;
                    for (i = l; i <= k; i++)
                    {
                        f = s * rv1[i];
                        rv1[i] = c * rv1[i];
                        if ((std::abs(f) + anorm) == anorm)
                            break;
                        g = w[i];
                        h = pythag(f, g);
                        w[i] = h;
                        h = 1.0 / h;
                        c = g * h;
                        s = -f * h;
                        for (j = 0; j < nRows; j++)
                        {
                            y = a[j][nm];
                            z = a[j][i];
                            a[j][nm] = y * c + z * s;
                            a[j][i] = z * c - y * s;
                        }
                    }
                }
                z = w[k];
                if (l == k)
                {
                    if (z < 0.0)
                    {
                        w[k] = -z;
                        for (j = 0; j < nCols; j++)
                            vt[k][j] = -vt[k][j];
                    }
                    break;
                }
                if (its == (max_its - 1))
                    std::cerr << "algoim::svd::SVD::svdcmp(): no convergence in " << max_its << "svdcmp iterations." << std::endl;
                x = w[l];
                nm = k - 1;
                y = w[nm];
                g = rv1[nm];
                h = rv1[k];
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                g = pythag(f, 1.0);
                f = ((x - z) * (x + z) + h * ((y / (f + detail::sign(g,f)))- h)) / x;
                c = s = 1.0;
                for (j = l; j <= nm; j++)
                {
                    i = j + 1;
                    g = rv1[i];
                    y = w[i];
                    h = s * g;
                    g = c * g;
                    z = pythag(f, h);
                    rv1[j] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = g * c - x * s;
                    h = y * s;
                    y *= c;
                    for (jj = 0; jj < nCols; jj++)
                    {
                        x = vt[j][jj];
                        z = vt[i][jj];
                        vt[j][jj] = x * c + z * s;
                        vt[i][jj] = z * c - x * s;
                    }
                    z = pythag(f, h);
                    w[j] = z;
                    if (z)
                    {
                        z = 1.0 / z;
                        c = f * z;
                        s = h * z;
                    }
                    f = c * g + s * y;
                    x = c * y - s * g;
                    for (jj = 0; jj < nRows; jj++)
                    {
                        y = a[jj][j];
                        z = a[jj][i];
                        a[jj][j] = y * c + z * s;
                        a[jj][i] = z * c - y * s;
                    }
                }
                rv1[l] = 0.0;
                rv1[k] = f;
                w[k] = x;
            }
        }

        return true;
    }

    // Transforms Bernstein coefficients into monomial coefficients.
    void monomialCoefficients(const real *bernstein, int P, real *monomial)
    {
        for (int i = 0; i < P; ++i)
            monomial[i] = 0.0;

        for (int i = 0; i < P; ++i)
        {
            for (int l = i; l < P; ++l)
                monomial[l] += bernstein[i] * pow(-1.0, l-i) * Binomial::c(P-1,l) * Binomial::c(l,i);
        }
    }

    // Transforms monomial coefficients into Bernstein coefficients.
    void bernsteinCoefficients(const real *monomial, int P, real *bernstein)
    {
        for (int i = 0; i < P; ++i)
            bernstein[i] = 0.0;

        for (int k = 0; k < P; ++k)
        {
            const real aux = monomial[k] / Binomial::c(P-1,k);
            for (int j = k; j < P; ++j)
                bernstein[j] += aux * Binomial::c(j,k);
        }
    }

    /**
     * @brief Divides the given Bernstein polynomial by x assuming
     * that its first coefficient (alpha[0]) is zero.
     * 
     * @param alpha Bernstein polynomial coefficients.
     * @param P Polynomial degree.
     * @param beta Resulting polynomial (of degree P-1).
     */
    static void divideByX(const real *alpha, const int P, real *beta)
    {
        real tol = 0.0;
        for (int i = 0; i < P; ++i)
            tol = std::max(tol, std::abs(alpha[i]));
        tol *= 1.0e4 * std::numeric_limits<real>::epsilon();
        assert(std::abs(alpha[0]) < tol && "First Bernstein coefficient must be zero.");

        // Transforming to Bernstein coefficients to monomial (gamma[k] x^k)
        real *gamma;
        algoim_spark_alloc(real, &gamma, P);
        monomialCoefficients(alpha, P, gamma);

        // Dividing by x (gamma[k] x^k-1, notice gamma[0] = 0) and transforming back to Bernstein
        bernsteinCoefficients(gamma+1, P-1, beta);
    }


    /**
     * @brief Finds all eigenvalues of an upper Hessenberg matrix a[0..n-1][0..n-1].
     * 
     * Adapted from Numerical recipes
     *  https://github.com/blackstonep/Numerical-Recipes/blob/master/eigen_unsym.h
     * 
     * On input a can be exactly as output from elmhes 11.6; on output it is destroyed.
     * The real part of the eigenvalues is returned in wri[0..n-1] and the imaginary one in wi[0..n-1].
     * wri stores the complex n eigenvalues. Real and imaginary parts are stored
     * contiguously, i.e., wri[i*2] and wri[i*2+1] are the real and imaginary parts, respectively,
     * of the i-th eigenvalue.
     */
    static bool hqr(real **a, int n, real *wri)
    {
        /// Maximum number of iterations.
        static const int max_its = 30;

        using std::abs;

        int nn,m,l,k,j,its,i,mmin;
        real z,y,x,w,v,u,t,s,r,q,p,anorm=0.0;

        const real EPS = std::numeric_limits<real>::epsilon();
        for (i=0;i<n;i++)
        {
            wri[2*i]=0.0;
            wri[2*i+1]=0.0;
            for (j=std::max(i-1,0);j<n;j++)
                anorm += abs(a[i][j]);
        }
        nn=n-1;
        t=0.0;
        while (nn >= 0)
        {
            its=0;
            do
            {
                for (l=nn;l>0;l--)
                {
                    s=abs(a[l-1][l-1])+abs(a[l][l]);
                    if (s == 0.0)
                        s=anorm;
                    if (abs(a[l][l-1]) <= EPS*s)
                    {
                        a[l][l-1] = 0.0;
                        break;
                    }
                }
                x=a[nn][nn];
                if (l == nn)
                {
                    // wri[nn--]=x+t;
                    wri[2*nn]=x+t;--nn;
                }
                else
                {
                    y=a[nn-1][nn-1];
                    w=a[nn][nn-1]*a[nn-1][nn];
                    if (l == nn-1)
                    {
                        p=0.5*(y-x);
                        q=p*p+w;
                        z=std::sqrt(abs(q));
                        x += t;
                        if (q >= 0.0)
                        {
                            z=p+detail::sign(z,p);
                            // wri[nn-1]=wri[nn]=x+z;
                            wri[2*(nn-1)]=wri[2*nn]=x+z;
                            if (z != 0.0)
                                // wri[nn]=x-w/z;
                                wri[2*nn]=x-w/z;
                        }
                        else
                        {
                            // wri[nn]=Complex(x+p,-z);
                            wri[2*nn]=x+p;
                            wri[2*nn+1]=-z;
                            // wri[nn-1]=conj(wri[nn]);
                            wri[2*(nn-1)]=wri[2*nn];
                            wri[2*(nn-1)+1]=-wri[2*nn+1];
                        }
                        nn -= 2;
                    }
                    else
                    {
                        if (its == max_its) // its == 30
                        {
                            // throw("Too many iterations in hqr");
                            return false;
                        }
                        if (0 < its && (its % 10) == 0) // its == 10 || its == 20
                        {
                            t += x;
                            for (i=0;i<nn+1;i++)
                                a[i][i] -= x;
                            s=abs(a[nn][nn-1])+abs(a[nn-1][nn-2]);
                            y=x=0.75*s;
                            w = -0.4375*s*s;
                        }
                        ++its;
                        for (m=nn-2;m>=l;m--)
                        {
                            z=a[m][m];
                            r=x-z;
                            s=y-z;
                            p=(r*s-w)/a[m+1][m]+a[m][m+1];
                            q=a[m+1][m+1]-z-r-s;
                            r=a[m+2][m+1];
                            s=abs(p)+abs(q)+abs(r);
                            p /= s;
                            q /= s;
                            r /= s;
                            if (m == l)
                                break;
                            u=abs(a[m][m-1])*(abs(q)+abs(r));
                            v=abs(p)*(abs(a[m-1][m-1])+abs(z)+abs(a[m+1][m+1]));
                            if (u <= EPS*v)
                                break;
                        }
                        for (i=m;i<nn-1;i++)
                        {
                            a[i+2][i]=0.0;
                            if (i != m)
                                a[i+2][i-1]=0.0;
                        }
                        for (k=m;k<nn;k++)
                        {
                            if (k != m)
                            {
                                p=a[k][k-1];
                                q=a[k+1][k-1];
                                r=0.0;
                                if (k+1 != nn)
                                    r=a[k+2][k-1];
                                if ((x=abs(p)+abs(q)+abs(r)) != 0.0)
                                {
                                    p /= x;
                                    q /= x;
                                    r /= x;
                                }
                            }
                            if ((s=sign(std::sqrt(p*p+q*q+r*r),p)) != 0.0)
                            {
                                if (k == m)
                                {
                                    if (l != m)
                                    a[k][k-1] = -a[k][k-1];
                                }
                                else
                                    a[k][k-1] = -s*x;
                                p += s;
                                x=p/s;
                                y=q/s;
                                z=r/s;
                                q /= p;
                                r /= p;
                                for (j=k;j<nn+1;j++)
                                {
                                    p=a[k][j]+q*a[k+1][j];
                                    if (k+1 != nn)
                                    {
                                        p += r*a[k+2][j];
                                        a[k+2][j] -= p*z;
                                    }
                                    a[k+1][j] -= p*y;
                                    a[k][j] -= p*x;
                                }
                                mmin = nn < k+3 ? nn : k+3;
                                for (i=l;i<mmin+1;i++)
                                {
                                    p=x*a[i][k]+y*a[i][k+1];
                                    if (k+1 != nn)
                                    {
                                        p += z*a[i][k+2];
                                        a[i][k+2] -= p*r;
                                    }
                                    a[i][k+1] -= p*q;
                                    a[i][k] -= p;
                                }
                            }
                        }
                    }
                }
            } while (l+1 < nn);
        }

        return true;
    }

    /**
     * @brief Given a row-wise upper Hessenberg matrix A with size n x n, it computes its (complex)
     * eigenvalues using the HQR algorithm.
     *     R. S. Martin, G. Peters, and J. H. Wilkinson.
     *     Handbook Series Linear Algebra: The QR algorithm for real Hessenberg matrices.
     *     Numer. Math., 14(3):219–231, 1970.
     * 
     * @param A row-wise upper Hessenberg  matrix A. This matrix is polluted when the function returns.
     * @param n Number of rows/cols of matrix A.
     * @param w Output vector storing the complex n eigenvalues. Real and imaginary parts are stored
     * contiguously, i.e., w[i*2] and w[i*2+1] are the real and imaginary parts, respectively, of the i-th eigenvalue.
     * @return True is successful, false otherwise.
     * 
     * @note w must be already allocated (size 2*n) when calling the function.
     */
    static bool hqr(real *A, int n, real *wri)
    {
        assert(0 < n);

        std::vector<real *> a(n);

        for(int i = 0; i < n; ++i)
            a[i] = A + i * n;

        return detail::hqr(a.data(), n, wri);
    }


#ifdef WITH_LAPACK
    /**
     * @brief Computes the generalized eigenvalues for the non-symmetric matrices A and B.
     * I.e., it computes lambda such that A x = lambda B x
     * 
     * @param A First non-symmetric matrix.
     * @param B Second non-symmetric matrix.
     * @param out Complex eigenvalues to be computed.
     * @return True is succeed, false otherwise.
     */
    bool generalisedEigenvaluesLAPACK(xarray<real,2>& A, xarray<real,2>& B, xarray<real,2>& out)
    {
        int N = A.ext(0);
        assert(all(A.ext() == N) && all(B.ext() == N) && out.ext(0) == N && out.ext(1) == 2);
        real *alphar, *alphai, *beta, *lscale, *rscale;
        algoim_spark_alloc(real, &alphar, N, &alphai, N, &beta, N, &lscale, N, &rscale, N);
        real abnrm, bbnrm;
        lapack_int ilo, ihi;
        static_assert(std::is_same_v<real, double>, "Algoim's default LAPACK code assumes real == double; a custom generalised eigenvalue solver is required when real != double");
        const int info = LAPACKE_dggevx(LAPACK_ROW_MAJOR, 'B', 'N', 'N', 'N', N, A.data(), N, B.data(), N, alphar, alphai, beta, nullptr, N, nullptr, N, &ilo, &ihi, lscale, rscale, &abnrm, &bbnrm, nullptr, nullptr);
        const bool success = info == 0;
        for (int i = 0; i < N; ++i)
        {
            if (beta[i] != 0.0)
                out(i,0) = alphar[i] / beta[i],
                out(i,1) = alphai[i] / beta[i];
            else
                out(i,0) = out(i,1) = std::numeric_limits<real>::infinity();
        }
        return success;
    }
#endif // WITH_LAPACK

} // namespace detail

    /**
     * @brief Compute all complex roots of a Bernstein polynomial.
     * 
     * This function applies the method in
     * G. F. Jónsson and S. Vavasis,
     * Solving polynomials with small leading coefficients,
     * SIAM Journal on Matrix Analysis and Applications, 26 (2004), pp. 400–414
     * https://doi.org/10.1137/S0895479899365720.
     * 
     * In the case no LAPACK support is provided, it modifies slightly the algorithm
     * as discussed before.
     * 
     * @param alpha Bernstein coefficients (array of length @p P).
     * @param P Order (degree + 1) of the polynomial.
     * @param out Complex eigenvalues to be computed (array of length (P-1) x 2).
     */
    static void rootsBernsteinPoly(const real* alpha, int P, xarray<real,2>& out)
    {
        assert(P >= 2 && out.ext(0) == P - 1 && out.ext(1) == 2);
        using std::max;
        using std::abs;

        real *beta;
        algoim_spark_alloc(real, &beta, P);
        real tol = 0.0;
        for (int i = 0; i < P; ++i)
            tol = max(tol, abs(alpha[i]));
        tol *= 1.0e4 * std::numeric_limits<real>::epsilon();
        for (int i = 0; i < P; ++i)
            beta[i] = (abs(alpha[i]) > tol) ? alpha[i] : 0;

        int N = P - 1;
        bool success{true};
#ifdef WITH_LAPACK
        // We solve the generalized eigenvalue problem A x = t B x,
        // with A and B defined in Jónsson and Vavasis 2004.
        xarray<real,2> A(nullptr, uvector<int,2>{N, N});
        xarray<real,2> B(nullptr, uvector<int,2>{N, N});
        algoim_spark_alloc(real, A, B);
        A = 0;
        B = 0;
        for (int i = 0; i < N - 1; ++i)
            A(i, i + 1) = B(i, i + 1) = 1.0;
        for (int i = 0; i < N; ++i)
            A(N - 1, i) = B(N - 1, i) = -beta[i];
        B(N - 1, N - 1) += beta[N] / N;
        for (int i = 0; i < N - 1; ++i)
            B(i, i) = real(N - i) / real(i + 1);

        success = detail::generalisedEigenvaluesLAPACK(A, B, out);
#else // WITH_LAPACK
        // Instead of solving the generalized eigenvalue problem A x = t B x,
        // we solve the eigenvalue problem r x = C x, where r = 1 / t.
        // Given the structure of A and B above, C = A^-1 B is a Hessenberg matrix easily computable.
        // Then, we compute the eigenvalues of C (non-symmetric) using the HQR algorithm.

        if (abs(beta[0]) < tol)
        {
            // beta[0] = 0, then, there is one root at x=0
            out(0, 0) = 0.0;
            out(0, 1) = 0.0;

            // Now, we compute the remaining roots from the polynomial p(x) / x
            real *gamma;
            algoim_spark_alloc(real, &gamma, P-1);
            detail::divideByX(beta, P, gamma);

            xarray<real,2> out2(out.data() + 2, uvector<int,2>{P - 2, 2});
            rootsBernsteinPoly(gamma, P-1, out2);
            return;
        }
        else
        {
            xarray<real,2> C(nullptr, uvector<int,2>{N, N});
            algoim_spark_alloc(real, C);
            C = 0;
            for (int i = 0; i < N; ++i)
            {
                C(i, i) = 1.0;
                C(0, i) -= beta[i+1] / beta[0] * real(N - i) / real(i + 1);
            }
            for (int i = 0; i < N - 1; ++i)
                C(i+1, i) = real(N - i) / real(i + 1);

            success = detail::hqr(C.data(), N, out.data());

            // Inverting eigenvalues (r = 1 / t).
            for (int i = 0; i < N; ++i)
            {
                const bool img = abs(out(i, 1)) > tol;
                if (img)
                    out(i, 1) = -out(i, 1);
                else
                    out(i, 1) = 0.0;

                const auto l = out(i, 0) * out(i, 0) + out(i, 1) * out(i, 1);

                if (abs(l) < tol)
                {
                    out(i, 0) = std::numeric_limits<real>::infinity();
                    if (img)
                        out(i, 1) = std::numeric_limits<real>::infinity();
                }
                else
                {
                    out(i, 0) /= l;
                    out(i, 1) /= l;
                }
            }
        }
#endif // WITH_LAPACK

        assert(success && "Eigenvalue calculation failed (algoim::eig::rootsBernsteinPoly)");
    }

    /**
     * @brief Given a row-wise matrix A with size m x n (m > n), it computes
     * its singular value decomposition, A = U * W * Vt.
     * 
     * A is replaced by U. The diagonal matrix W is output as a vector
     * of length n, and V (not V transpose) is output as row-wise matrix of size n x n.
     * 
     * @param A row-wise matrix A to decompose with size m x n (m > n).
     * When the function returns, the content is destroyed.
     * @param m Number of rows of matrix A.
     * @param n Number of columns of matrix A.
     * @param U is output as row-wise matrix of size m x m.
     * @param W is output diagonal matrix as min(m,n)-sized vector.
     * @param Vt is output as row-wise matrix of size n x n.
     * 
     * @note U, W, and V must be already allocated when calling the function.
     * @note If the this function is compiled with WITH_LAPACK, LAPACK's dgesvd is called,
     * otherwise, a SVD version from Numerical Recipes implemented in the library.
     */
    static void computeSVD(real * const A, const int m, const int n, real * const U, real * const W, real * const Vt)
    {
        assert(0 < m && 0 < n);

#ifdef WITH_LAPACK

        static_assert(std::is_same_v<real, double>, "Algoim's default LAPACK code assumes real == double; a custom SVD solver is required when real != double");

        real *superb;
        const int l_superb = std::max(1, std::max(3*std::min(m,n)+std::max(m,n), 5*std::min(m,n)));
        algoim_spark_alloc(real, &superb, l_superb);

        const auto info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, n, W, U, m, Vt, n, superb);
        const bool success = info == 0;

#else // WITH_LAPACK

        assert(n <= m && "Not implemented.");

        std::vector<real *> u(m);
        std::vector<real *> vt(n);

        for(int i = 0; i < m; ++i)
        {
            u[i] = U + i * m;

            for(int j = 0; j < n; ++j)
                u[i][j] = A[i * n + j];
            for(int j = n; j < m; ++j)
                u[i][j] = 0.0;
        }

        for(int i = 0; i < n; ++i)
            vt[i] = Vt + i * n;

        const auto success = detail::svdcmp(u.data(), m, n, W, vt.data());

#endif // WITH_LAPACK

        assert(success && "SVD call failed (algoim::eig:svd)");
    }

} // namespace algoim::bernstein

#endif // ALGOIM_EIG_UTILS_HPP

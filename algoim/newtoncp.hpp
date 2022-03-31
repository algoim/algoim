#ifndef ALGOIM_NEWTONCP_HPP
#define ALGOIM_NEWTONCP_HPP

/* Newton's method for the constrained minimum-distance optimisation problem applied to
   finding closest points on the zero level set of a function (typically a polynomial)

   For more information, refer to the paper
      R. I. Saye, High-order methods for computing distances to implicitly defined surfaces,
      Communications in Applied Mathematics and Computational Science, 9(1), 107-141 (2014),
      http://dx.doi.org/10.2140/camcos.2014.9.107
*/

#include "uvector.hpp"
#include "utility.hpp"

namespace algoim
{
    namespace detail
    {
        // Gaussian elimination with partial pivoting, solving the system Ax = b, such that
        // b is overwritten with the solution x, and A is overwritten with LU decomposition
        // (modulo permutations). Returns false if and only if a "small" pivot is detected,
        // assuming largest singular value is O(1).
        template<int N>
        bool newtoncp_gepp(uvector<double,N*N>& A, uvector<double,N>& b)
        {
            for (int i = 0; i < N; ++i)
            {
                int j = i;
                for (int k = i + 1; k < N; ++k)
                    if (std::abs(A(k*N+i)) > std::abs(A(j*N+i)))
                        j = k;
                if (j != i)
                {
                    for (int k = 0; k < N; ++k)
                        std::swap(A(i*N+k), A(j*N+k));
                    std::swap(b(i), b(j));
                }

                if (std::abs(A(i*N+i)) < 1.0e4*std::numeric_limits<double>::epsilon())
                    return false;

                double fac = 1.0 / A(i*N+i);
                for (int j = i + 1; j < N; ++j)
                    A(j*N+i) *= fac;

                for (int j = i + 1; j < N; ++j)
                {
                    for (int k = i + 1; k < N; ++k)
                        A(j*N+k) -= A(j*N+i)*A(i*N+k);
                    b(j) -= A(j*N+i)*b(i);
                }
            }

            for (int i = N - 1; i >= 0; --i)
            {
                double sum = 0.0;
                for (int j = i + 1; j < N; ++j)
                    sum += A(i*N+j)*b(j);
                b(i) = (b(i) - sum) / A(i*N+i);
            }

            return true;
        }
    }

    /* Newton's method for the constrained minimum-distance optimisation problem:
        - x: initial guess of the closest point
        - ref: query point for which argmin ||x - ref|| is sought
        - phi: level set function whose zero level set defines the surface
        - r: radius of the bounding ball
        - tolsqr: squared tolerance that determines convergence criterion
        - maxsteps: maximum number of steps in the iterative method
    */
    template<int N, typename F>
    int newtonCP(uvector<double,N>& x, const uvector<double,N>& ref, const F& phi, double r, double tolsqr, int maxsteps)
    {
        uvector<double,N> x0 = x;
        double lambda = 0.0;
        for (int step = 1; step <= maxsteps; ++step)
        {
            uvector<double,N> xold = x;

            // Evaluate phi and its derivatives
            double phival = phi(x);
            uvector<double,N> phigrad = phi.grad(x);
            uvector<double,N*(N+1)/2> phiHessian = phi.hessian(x);

            // Since x is on or very near the zero level set of phi, it is unlikely that |nabla(phi)| = 0. If it is,
            // then the closest point problem is ill-posed, so terminate with x being the closest point.
            double magsqrgrad = sqrnorm(phigrad);
            if (magsqrgrad < 1e-4*tolsqr)
                return step;

            // Initialise lambda at step 0 assuming near closest point
            if (step == 1)
                lambda = dot(ref - x, phigrad) / magsqrgrad;

            // Calculate gradient of functional
            uvector<double,N+1> gradf;
            for (int i = 0; i < N; ++i)
                gradf(i) = x(i) - ref(i) + lambda*phigrad(i);
            gradf(N) = phival;

            // Calculate Hessian of functional
            uvector<double,(N+1)*(N+1)> Hf;
            int k = 0;
            for (int i = 0; i < N; ++i)
            {
                Hf(i*(N+1)+i) = 1.0 + lambda*phiHessian(k++);
                for (int j = i + 1; j < N; ++j)
                {
                    double Hij = lambda*phiHessian(k++);
                    Hf(i*(N+1)+j) = Hij;
                    Hf(j*(N+1)+i) = Hij;
                }
                Hf(i*(N+1)+N) = phigrad(i);
                Hf(N*(N+1)+i) = phigrad(i);
            }
            Hf(N*(N+1)+N) = 0.0;

            // Apply Newton's method
            if (detail::newtoncp_gepp(Hf, gradf))
            {
                // Clamp update to ensure do not move too far
                double msqr = 0.0;
                for (int i = 0; i < N; ++i)
                    msqr += util::sqr(gradf(i));
                if (msqr > util::sqr(0.5*r))
                    gradf *= 0.5*r/sqrt(msqr);

                // Update
                for (int i = 0; i < N; ++i)
                    x(i) -= gradf(i);
                lambda -= gradf(N);
            }
            else
            {
                // Newton's method failed, since Hessian was detected to be approximately singular. This generally
                // only occurs when x is (very) near the centre of curvature of the surface. In this case, revert to 
                // a type of gradient descent
                uvector<double,N> delta1 = (phival/magsqrgrad)*phigrad;
                lambda = dot(ref - x, phigrad) / magsqrgrad;
                uvector<double,N> delta2 = x - ref + lambda*phigrad;
                // Clamp delta2, the tangential direction, necessary when interface undergoes high curvature
                double msqr = sqrnorm(delta2);
                if (msqr > util::sqr(0.1*r))
                    delta2 *= 0.1*r/sqrt(msqr);
                x -= delta1 + delta2;
            }

            if (sqrnorm(x - x0) > util::sqr(r))
            {
                // Restore x to the last iterate inside the bounding ball
                x = xold;
                return -1;
            }

            if (sqrnorm(x - xold) < tolsqr)
			    return step;
        }
        return -2;
    }

    // Simple Newton-style procedure for projecting a point x onto the zero level set of f
    template<int N, typename F>
    int newtonIso(uvector<double,N>& x, const F& f, double tolsqr, int maxsteps)
    {
        for (int step = 1; step <= maxsteps; ++step)
        {
            // Evaluate f and its gradient
            double val = f(x);
            uvector<double,N> g = f.grad(x);

            // Compute delta and update
            double msqr = sqrnorm(g);
            if (msqr > 0.0)
                g *= -val/msqr;
            x += g;

            // Terminate if converged to within tolerance
            if (sqrnorm(g) < tolsqr)
                return step;
        }
        return -1;
    }
} // namespace algoim

#endif
